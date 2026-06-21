import sys
import os
import time
import tempfile
import math
import numpy as np
import soundfile as sf
from scipy import signal

from PyQt6.QtCore import (Qt, QThread, pyqtSignal, QTimer, QRectF, QPointF, QUrl)
from PyQt6.QtGui import (QColor, QPainter, QPen, QFont, QPainterPath,
                         QLinearGradient, QBrush, QRadialGradient, QIcon, QPixmap)
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QFrame,
                             QFileDialog, QSizePolicy, QGraphicsOpacityEffect)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

SR = 44100

# ====================== DRUM ENGINES ======================

class SynthEngine:
    KIT_SIMPLE, KIT_PCM, KIT_EIGHT, KIT_NINE = 0, 1, 2, 3
    current_kit = KIT_EIGHT

    @staticmethod
    def set_kit(i):
        SynthEngine.current_kit = i % 4

    @staticmethod
    def finalize(data):
        if len(data) < 200:
            return data.astype(np.float32)
        fade = min(50, len(data) // 4)
        if fade > 0:
            data[-fade:] *= np.linspace(1, 0, fade)
        sos = signal.butter(1, 18000, 'lp', fs=SR, output='sos')
        data = signal.sosfilt(sos, data)
        peak = np.max(np.abs(data))
        if peak > 0:
            data /= peak
        return SynthEngine.ensure_zero_crossing(data.astype(np.float32))

    @staticmethod
    def ensure_zero_crossing(data):
        if len(data) < 200:
            return data
        limit = min(len(data) // 4, 2000)
        zc = np.where(np.diff(np.sign(data[:limit])))[0]
        if len(zc) > 0:
            return data[zc[0] + 1:]
        return data

    @staticmethod
    def apply_filter(data, val):
        if 0.45 < val < 0.55:
            return data
        if val <= 0.45:
            cutoff = 150 + ((val / 0.45) ** 2 * 18000)
            sos = signal.butter(2, cutoff, 'lp', fs=SR, output='sos')
        else:
            cutoff = 20 + (((val - 0.55) / 0.45) ** 2 * 8000)
            sos = signal.butter(2, cutoff, 'hp', fs=SR, output='sos')
        return signal.sosfilt(sos, data).astype(np.float32)

    @staticmethod
    def resample_lofi(data, crush):
        if crush <= 0.01:
            return data
        reduction = 1.0 + (crush * 5.0)
        orig = len(data)
        tgt = max(1, int(orig / reduction))
        lo = signal.resample(data, tgt)
        bits = 16 - (crush * 8)
        steps = 2 ** bits
        lo = np.round(lo * steps) / steps
        restored = signal.resample(lo, orig).astype(np.float32)
        return SynthEngine.ensure_zero_crossing(np.clip(restored, -1, 1))

    @staticmethod
    def process_sample(raw, params):
        if raw is None or len(raw) == 0:
            return np.zeros(100, dtype=np.float32)
        speed = 0.5 + (params.get('pitch', 0.5) * 1.5)
        nl = max(10, int(len(raw) / speed))
        y = signal.resample(raw, nl)
        t = np.linspace(0, len(y) / SR, len(y))
        tone = params.get('tone', 0.5)
        if tone < 0.45:
            sos = signal.butter(1, 500 + tone * 8000, 'lp', fs=SR, output='sos')
            y = signal.sosfilt(sos, y)
        elif tone > 0.55:
            sos = signal.butter(1, 100 + (tone - 0.5) * 4000, 'hp', fs=SR, output='sos')
            y = signal.sosfilt(sos, y)
        decay = 0.5 + ((1.0 - params.get('decay', 0.5)) * 15)
        return SynthEngine.finalize(y * np.exp(-t * decay))

    @staticmethod
    def generate_drum(dtype, params):
        try:
            kits = [SimpleDrums, PCMDrums, EightDrums, NineDrums]
            audio = kits[SynthEngine.current_kit].generate(dtype, params)
            if audio is None or len(audio) == 0:
                return np.zeros(1024, dtype=np.float32)
            return audio
        except Exception as e:
            print(f"drum gen err: {e}")
            return np.zeros(1024, dtype=np.float32)

    @staticmethod
    def apply_background_reverb(audio):
        if len(audio) == 0:
            return audio
        wet = np.zeros_like(audio)
        for d_s, f_cut, amp in [(0.04, None, 0.5), (0.12, 3500, 0.35), (0.25, 1500, 0.20)]:
            d = int(SR * d_s)
            if d < len(audio):
                r = np.roll(audio, d)
                r[:d] = 0
                if f_cut:
                    r = signal.sosfilt(signal.butter(1, f_cut, 'lp', fs=SR, output='sos'), r)
                wet += r * amp
        wet = signal.sosfilt(signal.butter(1, 300, 'hp', fs=SR, output='sos'), wet)
        return audio + wet * 0.20


class SimpleDrums:
    @staticmethod
    def generate(dt, p):
        pp, pd, pt = p.get('pitch', 0.5), p.get('decay', 0.5), p.get('tone', 0.5)
        dur = 0.6
        t = np.linspace(0, dur, int(SR * dur), endpoint=False)
        y = np.zeros_like(t)
        if dt == "kick":
            fs = 120 + pp * 50; fe = 40
            fe_env = (fs - fe) * np.exp(-t * 12) + fe
            ph = np.cumsum(fe_env) * 2 * np.pi / SR
            osc = np.sin(ph)
            sos = signal.butter(1, [200, 350], 'bs', fs=SR, output='sos')
            osc = signal.sosfilt(sos, osc)
            y = signal.sosfilt(signal.butter(2, 5000, 'lp', fs=SR, output='sos'), osc) * np.exp(-t * (6 + (1 - pd) * 40))
        elif dt == "snare":
            fb = 150 + pp * 25
            body = np.sin(2 * np.pi * fb * t) * np.exp(-t * 20)
            n = np.random.uniform(-0.5, 0.5, len(t))
            fc = 2200 + pt * 1500
            n = signal.sosfilt(signal.butter(2, [fc - 1000, fc + 1000], 'bp', fs=SR, output='sos'), n)
            body = signal.sosfilt(signal.butter(1, [400, 700], 'bs', fs=SR, output='sos'), body)
            y = (body * 0.6) + (n * np.exp(-t * (15 + (1 - pd) * 40)) * 0.45)
        elif "hat" in dt:
            n = np.random.uniform(-1, 1, len(t))
            bc = 7000 + pt * 3000
            hiss = signal.sosfilt(signal.butter(2, [bc - 2000, bc + 2000], 'bp', fs=SR, output='sos'), n)
            fm = 800 + pp * 300
            metal = np.sin(2 * np.pi * (fm * 3.5) * t) * 0.2
            sig = hiss + metal
            if "closed" in dt:
                y = sig * np.minimum(t * 2000, 1) * np.exp(-t * (60 + (1 - pd) * 200))
            else:
                y = sig * np.minimum(t * 500, 1) * np.exp(-t * (10 + (1 - pd) * 30)) * 0.8
        elif dt == "clap":
            n = np.random.uniform(-1, 1, len(t))
            bl = 900 + pp * 200; bh = bl + 800
            f = signal.sosfilt(signal.butter(2, [bl, bh], 'bp', fs=SR, output='sos'), n)
            env = np.exp(-t * (10 + (1 - pd) * 30))
            a = min(len(t), int(SR * 0.015))
            env[:a] *= np.linspace(0, 1, a)
            y = f * env
        elif dt == "wood" or ("perc" in dt and "a" in dt):
            fb = 600 + pp * 300
            fe = fb * (1 + 0.1 * np.exp(-t * 50))
            ph = np.cumsum(fe) * 2 * np.pi / SR
            y = np.sin(ph) * np.exp(-t * (30 + (1 - pd) * 100))
        else:
            f = 125 + pp * 100
            fe = f * (1 - 0.2 * np.exp(-t * 10))
            ph = np.cumsum(fe) * 2 * np.pi / SR
            y = np.sin(ph) * np.exp(-t * (8 + (1 - pd) * 25))
        return SynthEngine.finalize(y)


class PCMDrums:
    @staticmethod
    def degrade(d, sr_t=22050, bd=8):
        if sr_t < SR:
            f = int(SR / sr_t)
            d = np.repeat(d[::f], f)[:len(d)]
        s = 2 ** bd
        return np.round(d * s) / s

    @staticmethod
    def generate(dt, p):
        pp, pd, pt = p.get('pitch', 0.5), p.get('decay', 0.5), p.get('tone', 0.5)
        dur = 0.5
        t = np.linspace(0, dur, int(SR * dur), endpoint=False)
        y = np.zeros_like(t)
        if dt == "kick":
            fb = 60 + pp * 20
            body = np.sin(2 * np.pi * fb * t)
            c = np.random.uniform(-1, 1, len(t))
            c = signal.sosfilt(signal.butter(2, 800, 'lp', fs=SR, output='sos'), c) * np.exp(-t * 100)
            y = (body * 0.8 + c * 0.4) * np.exp(-t * (10 + (1 - pd) * 30))
            y = PCMDrums.degrade(y, 16000, 10)
        elif dt == "snare":
            ft = 160 + pp * 40
            tone = np.sign(np.sin(2 * np.pi * ft * t))
            tone = signal.sosfilt(signal.butter(1, 400, 'lp', fs=SR, output='sos'), tone)
            n = np.random.uniform(-1, 1, len(t))
            n = signal.sosfilt(signal.butter(2, [600, 2000], 'bp', fs=SR, output='sos'), n)
            y = (tone * 0.4 + n * 0.8) * np.exp(-t * (15 + (1 - pd) * 50))
            y = PCMDrums.degrade(y, 24000, 8)
        elif "hat" in dt:
            n = np.random.uniform(-1, 1, len(t))
            hf = 6000 + pp * 2000
            y = signal.sosfilt(signal.butter(2, hf, 'hp', fs=SR, output='sos'), n)
            dv = 80 if "closed" in dt else 15
            dv += (1 - pd) * 100
            y *= np.exp(-t * dv)
            y = PCMDrums.degrade(y, 32000, 6)
        elif dt == "clap":
            n = np.random.uniform(-1, 1, len(t))
            bl = 600 + pp * 200
            f = signal.sosfilt(signal.butter(2, [bl, 3000], 'bp', fs=SR, output='sos'), n)
            dv = 10 + (1 - pd) * 30
            env = np.exp(-t * dv)
            di = int(0.015 * SR)
            env[di:] += 0.6 * np.exp(-t[:-di] * dv)
            y = f * env
            y = PCMDrums.degrade(y, 14000, 8)
        elif dt == "wood" or ("perc" in dt and "a" in dt):
            f = 500 + pp * 300
            y = np.sign(np.sin(2 * np.pi * f * t))
            y = signal.sosfilt(signal.butter(2, [f, f + 1500], 'bp', fs=SR, output='sos'), y)
            y *= np.exp(-t * (30 + (1 - pd) * 250))
            y = PCMDrums.degrade(y, 18000, 8)
        else:
            f = 100 + pp * 80
            y = np.sin(2 * np.pi * f * t) + np.random.uniform(-0.1, 0.1, len(t))
            y *= np.exp(-t * (8 + (1 - pd) * 20))
            y = PCMDrums.degrade(y, 12000, 9)
        return SynthEngine.finalize(y)


class EightDrums:
    @staticmethod
    def generate(dt, p):
        pp, pd, pt = p.get('pitch', 0.5), p.get('decay', 0.5), p.get('tone', 0.5)
        dur = 0.8
        t = np.linspace(0, dur, int(SR * dur), endpoint=False)
        y = np.zeros_like(t)
        rng = np.random.default_rng(int(pp * 1000 + pt * 100))
        if dt == "kick":
            fs = 80 + pp * 320; fe = 40 + pp * 10
            fe_env = (fs - fe) * np.exp(-t / 0.008) + fe
            ph = np.cumsum(fe_env) * 2 * np.pi / SR
            y = np.sin(ph)
            tc = 200 + pt * 4000
            thd = signal.sosfilt(signal.butter(2, tc, 'lp', fs=SR, output='sos'), rng.uniform(-1, 1, len(t))) * np.exp(-t * 150)
            y = (y + thd * 0.4) * np.exp(-t * (30 - pd * 28.5))
            if len(y) > 100:
                y[-100:] *= np.linspace(1, 0, 100)
        elif dt == "snare":
            fb = 140 + pp * 120
            to = np.sin(2 * np.pi * fb * t) * np.exp(-t * (30 + (1 - pd) * 60))
            fc = 1000 + pt * 5000
            n = rng.uniform(-1, 1, len(t))
            n = signal.sosfilt(signal.butter(2, [fc, fc + 2000], 'bp', fs=SR, output='sos'), n) * np.exp(-t * (30 + (1 - pd) * 80))
            y = (to * 0.5) + (n * 0.8)
        elif "hat" in dt:
            bf = 300 + pp * 200
            ratios = [2.0, 3.0, 4.16, 5.43, 6.79, 8.21]
            ms = np.zeros_like(t)
            for r in ratios:
                ms += np.sign(np.sin(2 * np.pi * bf * r * t + np.random.rand() * 2 * np.pi))
            ms /= len(ratios)
            ns = signal.sosfilt(signal.butter(2, 7000, 'hp', fs=SR, output='sos'), rng.uniform(-1, 1, len(t)))
            mr = 0.35 - pt * 0.1
            sig = (ms * mr) + (ns * (1 - mr))
            hf = 2000 + pt * 3000
            sig = signal.sosfilt(signal.butter(4, hf, 'hp', fs=SR, output='sos'), sig)
            bl = 6000 + pt * 1000
            sig = signal.sosfilt(signal.butter(2, [bl, bl + 8000], 'bp', fs=SR, output='sos'), sig)
            a = int(SR * 0.003)
            if len(sig) > a:
                sig[:a] *= 0.5 * (1 - np.cos(np.linspace(0, np.pi, a)))
            if "closed" in dt:
                y = sig * np.exp(-t * (90 + (0.75 - pd) * 250))
            else:
                y = sig * (0.7 * np.exp(-t * (10 + (1 - pd) * 50)) + 0.3 * np.exp(-t * (2 + (1 - pd) * 10)))
        elif dt == "clap":
            n = rng.uniform(-1, 1, len(t))
            lo = 900 + pp * 200; hi = 2500 + pp * 600
            f = signal.sosfilt(signal.butter(2, [lo, hi], 'bp', fs=SR, output='sos'), n)
            env = np.zeros_like(t); ps = 0.009
            td = 30 + (1 - pd) * 60
            for i in range(4):
                si = int(i * ps * SR)
                if si >= len(env):
                    break
                amp = 0.7 if i < 3 else 1.0
                rem = len(env) - si
                lt = np.linspace(0, rem / SR, rem)
                env[si:] = np.maximum(env[si:], np.exp(-lt * (250 if i < 3 else td)) * amp)
            y = f * env
        elif dt == "perc a":
            f = 400 + pp * 800
            fm = np.sin(2 * np.pi * (f * 0.5) * t) * (pt * 500)
            y = np.sin(2 * np.pi * (f + fm) * t) * np.exp(-t * (350 + (1 - pd) * 200))
        elif dt == "perc b":
            base = 60 + pp * 120; pe = 20 + pd * 40
            freq = base * (1 + (0.5 + pt) * np.exp(-t * pe))
            ph = np.cumsum(freq) * 2 * np.pi / SR
            y = np.tanh(np.sin(ph) * 1.2) * np.exp(-t * (20 + (1 - pd) * 80))
        y = signal.sosfilt(signal.butter(1, 17500, 'lp', fs=SR, output='sos'), y)
        peak = np.max(np.abs(y))
        if peak > 0:
            y /= peak
        if "hat" in dt:
            y *= 0.85
        return SynthEngine.finalize(y)


class NineDrums:
    @staticmethod
    def generate(dt, p):
        pp, pd, pt = p.get('pitch', 0.5), p.get('decay', 0.5), p.get('tone', 0.5)
        dur = 0.6
        t = np.linspace(0, dur, int(SR * dur), endpoint=False)
        y = np.zeros_like(t)
        if dt == "kick":
            fs = 150 + pp * 250; fe = 45 + pp * 30; fd = 30 + pd * 50
            fe_env = (fs - fe) * np.exp(-t * fd) + fe
            ph = np.cumsum(fe_env) * 2 * np.pi / SR
            osc = np.tanh(np.sin(ph) * (1 + pt * 4))
            y = (osc + np.random.normal(0, 0.5, len(t)) * np.exp(-t * 300) * 0.4) * np.exp(-t * (5 + (1 - pd) * 45))
        elif dt == "snare":
            n = np.random.uniform(-1, 1, len(t))
            sl = 1500 + pt * 1000
            snap = signal.sosfilt(signal.butter(2, [sl, sl + 6000], 'bp', fs=SR, output='sos'), n)
            fr = 180 + pp * 40
            tone = np.sin(np.cumsum(fr * (1 - 0.1 * t)) * 2 * np.pi / SR)
            y = (snap * np.exp(-t * (20 + (1 - pd) * 50)) * 0.9) + (tone * np.exp(-t * (25 + (1 - pd) * 30)) * 0.45)
            y = np.tanh(y * 1.1)
            y = signal.sosfilt(signal.butter(1, [350, 600], 'bs', fs=SR, output='sos'), y)
        elif "hat" in dt:
            sig = np.zeros_like(t)
            for f in [263, 400, 421, 474, 587, 845]:
                sig += np.sign(np.sin(2 * np.pi * f * (1 + pp * 0.2) * t))
            if "closed" in dt:
                hf = 7000 + pt * 2000
                sig = signal.sosfilt(signal.butter(4, hf, 'hp', fs=SR, output='sos'), sig)
                y = sig * np.minimum(t * 1500, 1) * np.exp(-t * (50 + (1 - pd) * 150)) * 0.6
            else:
                hf = 6000 + pt * 2000
                sig = signal.sosfilt(signal.butter(4, hf, 'hp', fs=SR, output='sos'), sig)
                y = np.tanh(sig * np.exp(-t * (10 + (1 - pd) * 35)) * 1.0) * 0.7
            y = signal.sosfilt(signal.butter(2, 13000, 'lp', fs=SR, output='sos'), y)
        elif dt == "clap":
            n = np.random.uniform(-1, 1, len(t))
            lo = 1000 + pp * 200; hi = 2400 + pp * 300
            f = signal.sosfilt(signal.butter(2, [lo, hi], 'bp', fs=SR, output='sos'), n)
            env = np.zeros_like(t); ps = 0.009
            for i in range(4):
                si = int(i * ps * SR)
                if si >= len(env):
                    break
                amp = 0.8 if i < 3 else 1.0
                rem = len(env) - si
                lt = np.linspace(0, rem / SR, rem)
                env[si:] = np.maximum(env[si:], np.exp(-lt * (300 if i < 3 else 20 + (1 - pd) * 50)) * amp)
            y = f * env
        elif dt == "wood" or ("perc" in dt and "a" in dt):
            c = 800 + pp * 400; mf = c * 2.41; mi = 3 * (1 - pt * 0.5)
            mod = np.sin(2 * np.pi * mf * t) * mi * c
            osc = np.sin(2 * np.pi * c * t + mod)
            y = osc * np.exp(-t * (80 + (1 - pd) * 200))
            y = signal.sosfilt(signal.butter(2, 400, 'hp', fs=SR, output='sos'), y)
        else:
            fb = 90 + pp * 60; sw = np.exp(-t * 15)
            fi = fb * (1 + 0.5 * sw)
            ph = np.cumsum(fi) * 2 * np.pi / SR
            y = np.tanh(np.sin(ph) * 1.1)
            cl = np.random.uniform(-1, 1, len(t)) * np.exp(-t * 400)
            y = (y + cl * 0.2) * np.exp(-t * (8 + (1 - pd) * 15))
        return SynthEngine.finalize(y)


# ====================== MASTER EFFECTS ======================

class Effects:
    @staticmethod
    def simple_reverb(x, sr, mix=0.3, room=0.8, damp=0.5):
        if mix <= 0:
            return x
        if x.ndim == 1:
            x = np.column_stack((x, x))
        mono = x.mean(axis=1)
        tl = int(sr * (1.5 + room * 2.0))
        px = np.pad(mono, (0, tl))
        n = np.random.randn(tl)
        env = np.exp(-np.linspace(0, 1, tl) * (4 + (1 - room) * 10))
        ir = signal.lfilter(*signal.butter(1, 0.2 * (1 - damp), btype='low'), n * env)
        wet = signal.fftconvolve(px, ir, mode='full')[:len(x)]
        wet = signal.sosfilt(signal.butter(1, 300, 'hp', fs=sr, output='sos'), wet)
        wet = wet / (np.max(np.abs(wet)) + 1e-9)
        ws = np.column_stack((wet, wet))
        return (1 - mix) * x + mix * ws

    @staticmethod
    def bitcrush(d, sr, depth=0.0):
        if depth <= 0:
            return d
        q = 2 ** (16 - depth * 8)
        d = np.round(d * q) / q
        rd = depth
        if rd > 0:
            step = int(1 + rd * 5)
            if d.ndim == 2:
                for c in range(2):
                    d[:, c] = np.repeat(d[::step, c], step)[:len(d)]
            else:
                d = np.repeat(d[::step], step)[:len(d)]
        return d

    @staticmethod
    def apply_tone(d, sr, val):
        if 0.48 < val < 0.52:
            return d
        y = d.copy()
        if val <= 0.5:
            f = min(100 * (200 ** (val * 2)), sr / 2 - 100)
            sos = signal.butter(2, f, 'low', fs=sr, output='sos')
        else:
            f = min(20 * (400 ** ((val - 0.5) * 2)), sr / 2 - 100)
            sos = signal.butter(2, f, 'high', fs=sr, output='sos')
        if y.ndim == 2:
            y[:, 0] = signal.sosfilt(sos, y[:, 0])
            y[:, 1] = signal.sosfilt(sos, y[:, 1])
        else:
            y = signal.sosfilt(sos, y)
        return y

    @staticmethod
    def apply_rand_filter(d, sr, intensity, bpm):
        if intensity <= 0.01:
            return d
        rng = np.random.default_rng()
        step = int((60 / bpm / 4) * sr)
        y = d.copy(); tl = len(y)
        for i in range(0, tl, step):
            if rng.random() > intensity:
                continue
            e = min(i + step, tl)
            c = y[i:e]
            ft = rng.choice(['lp', 'hp', 'bp'])
            if ft == 'lp':
                sos = signal.butter(2, rng.uniform(300, 1200), 'low', fs=sr, output='sos')
            elif ft == 'hp':
                sos = signal.butter(2, rng.uniform(2000, 5000), 'high', fs=sr, output='sos')
            else:
                ce = rng.uniform(400, 3000)
                w = 0.4 if intensity > 0.8 else 0.8
                sos = signal.butter(2, [ce * (1 - w / 2), ce * (1 + w / 2)], 'band', fs=sr, output='sos')
            if c.ndim == 2:
                c[:, 0] = signal.sosfilt(sos, c[:, 0]); c[:, 1] = signal.sosfilt(sos, c[:, 1])
            else:
                c = signal.sosfilt(sos, c)
            y[i:e] = c
        return y

    @staticmethod
    def apply_vol_pan(d, sr, intensity, bpm):
        if intensity <= 0.01:
            return d
        rng = np.random.default_rng()
        step = int((60 / bpm / 4) * sr)
        if d.ndim == 1:
            l = d.copy(); r = d.copy()
        else:
            l = d[:, 0].copy(); r = d[:, 1].copy()
        tl = len(l)
        for i in range(0, tl, step):
            e = min(i + step, tl)
            vd = rng.uniform(0, 0.6) * intensity
            v = 1 - vd
            pw = 0.9 * intensity
            pan = rng.uniform(-pw, pw)
            pa = (pan + 1) * (np.pi / 4)
            l[i:e] *= np.cos(pa) * v
            r[i:e] *= np.sin(pa) * v
        return np.column_stack((l, r))

    @staticmethod
    def apply_samplerate(d, osr, tsr):
        if tsr >= osr:
            return d
        tsr = max(1000, tsr)
        ns = int(len(d) * (tsr / osr))
        if ns < 2:
            return d
        if d.ndim == 2:
            l = signal.resample(signal.resample(d[:, 0], ns), len(d))
            r = signal.resample(signal.resample(d[:, 1], ns), len(d))
            return np.column_stack((l, r))
        return signal.resample(signal.resample(d, ns), len(d))


# ====================== AUDIO MIXER ======================

class AudioMixer:
    @staticmethod
    def mix(slots, bpm, swing, clip, rev_prob, steps=16):
        sb = 60.0 / bpm
        ss = sb / 4.0
        total = int(ss * steps * SR)
        if total % 2 != 0:
            total += 1
        so = int(ss * swing * 0.33 * SR)
        out = np.zeros(total + int(SR * 0.5), dtype=np.float32)
        se = np.ones_like(out)
        for s in slots:
            if "kick" in s.get('label', ''):
                pat = s['pattern']
                dl = int(SR * 0.12); al = int(SR * 0.005)
                ds = np.ones(dl, dtype=np.float32)
                if dl > al:
                    ds[:al] = np.linspace(1, 0, al)
                    ds[al:] = np.linspace(0, 1, dl - al)
                for i in range(steps):
                    if i < len(pat) and pat[i]:
                        st = int(i * ss * SR)
                        if i % 2 != 0:
                            st += so
                        if st < len(se):
                            w = min(len(ds), len(se) - st)
                            se[st:st + w] *= ds[:w]
        for s in slots:
            try:
                raw = s['data']
                if raw is None or len(raw) == 0:
                    continue
                raw = np.nan_to_num(raw, copy=False)
                is_sl = s.get('is_sliced', False)
                is_bass = s.get('is_bass', False)
                csw = 0 if is_sl else so
                track = np.zeros_like(out)
                if is_bass:
                    w = min(len(raw), len(track))
                    seg = raw[:w].copy()
                    seg *= se[:w]
                    track[:w] += seg * 0.6
                else:
                    pat = s['pattern']; vel = s['velocities']
                    if not is_sl:
                        msl = total + int(SR * 0.5)
                        if clip > 0:
                            kr = 1.0 / (1 + clip * 20)
                            al = max(150, int(len(raw) * kr))
                            df = raw[:al].copy()
                            fs = min(200, int(al * 0.4))
                            if fs > 0:
                                df[-fs:] *= np.linspace(1, 0, fs)
                        else:
                            df = raw[:min(len(raw), msl)].copy()
                        fi = min(100, len(df) // 10)
                        if fi > 0:
                            df[:fi] *= np.linspace(0, 1, fi)
                        dr = df[::-1] if rev_prob > 0 else df
                        sl = len(df); ik = "kick" in s.get('label', '')
                        for i in range(steps):
                            if i >= len(pat):
                                break
                            if pat[i]:
                                sp = int(i * ss * SR)
                                if i % 2 != 0:
                                    sp += csw
                                if sp < len(track):
                                    cur = dr if (rev_prob > 0 and np.random.random() < rev_prob) else df
                                    w = min(sl, len(track) - sp)
                                    if w > 0:
                                        g = (vel[i] ** 1.5) * (1.0 if ik else 0.65)
                                        track[sp:sp + w] += cur[:w] * g
                    else:
                        ssl = len(raw) // steps
                        if ssl < 100:
                            continue
                        for i in range(steps):
                            if i >= len(pat):
                                break
                            if pat[i]:
                                ssr = i * ssl; sse = ssr + ssl
                                ds2 = int(i * ss * SR)
                                if i % 2 != 0:
                                    ds2 += csw
                                if sse > len(raw):
                                    sse = len(raw)
                                if ds2 >= len(track):
                                    continue
                                ch = raw[ssr:sse].copy()
                                ch *= (vel[i] ** 1.5)
                                fl = min(200, int(len(ch) * 0.05))
                                if fl > 4:
                                    ch[:fl] *= np.linspace(0, 1, fl)
                                    ch[-fl:] *= np.linspace(1, 0, fl)
                                w = min(len(ch), len(track) - ds2)
                                if w > 0:
                                    track[ds2:ds2 + w] += ch[:w] * 0.65
                if is_sl:
                    track *= se
                    track = SynthEngine.apply_background_reverb(track)
                out += track
            except:
                continue
        tail = out[total:]
        wl = min(len(tail), total)
        out[:wl] += tail[:wl]
        final = out[:total]
        peak = np.max(np.abs(final))
        if peak > 0.95:
            final = np.tanh(final) * 0.95
        return final


# ====================== RESEQ ENGINE ======================

class ReseqEngine:
    @staticmethod
    def process(raw, bpm, params, steps=16):
        if raw is None or len(raw) == 0:
            return np.zeros(int(SR * 2), dtype=np.float32)
        bars = max(1, steps // 16)
        tl = int((60 / bpm) * 4 * bars * SR)
        w = raw.astype(np.float32)
        speed = 0.5 + (params.get('pitch', 0.5) * 1.5)
        if abs(speed - 1) > 0.001:
            w = np.interp(np.linspace(0, len(w) - 1, int(len(w) / speed)), np.arange(len(w)), w).astype(np.float32)
        pt = params.get('tone', 0.5)
        if abs(pt - 0.5) > 0.05:
            if pt < 0.5:
                w = signal.sosfilt(signal.butter(1, 400 + pt * 10000, 'lp', fs=SR, output='sos'), w)
            else:
                w = signal.sosfilt(signal.butter(1, 100 + (pt - 0.5) * 5000, 'hp', fs=SR, output='sos'), w)
        if len(w) < tl:
            fe = min(50, len(w) // 8)
            if fe > 0:
                w[:fe] *= np.linspace(0, 1, fe); w[-fe:] *= np.linspace(1, 0, fe)
            m = np.concatenate([w, w[::-1]])
            w = np.tile(m, (tl // len(m)) + 2)
        w = w[:tl]
        sf_len = 64
        w[:sf_len] *= np.linspace(0, 1, sf_len)
        w[-sf_len:] *= np.linspace(1, 0, sf_len)
        w = SynthEngine.apply_filter(w, params.get('filter', 0.5))
        pc = params.get('crush', 0)
        if pc > 0.01:
            w = SynthEngine.resample_lofi(w, pc)
        pd = params.get('decay', 0.5)
        if pd < 0.4:
            gl = tl // steps
            ge = np.ones(gl, dtype=np.float32)
            tr = 0.9 - pd * 2
            tail = int(gl * tr)
            if tail > 0:
                ge[-tail:] = np.linspace(1, 0, tail)
            fg = np.tile(ge, steps)
            if len(fg) > len(w):
                fg = fg[:len(w)]
            elif len(fg) < len(w):
                fg = np.pad(fg, (0, len(w) - len(fg)), constant_values=1)
            w *= fg
        pk = np.max(np.abs(w))
        if pk > 0.01:
            w *= 0.6 / pk
        return w


# ====================== PATTERN GEN ======================

class PatternGen:
    @staticmethod
    def build(track_type, dens, steps=16):
        pat = [False] * steps
        if track_type == "kick":
            for i in range(0, steps, 4):
                pat[i] = True
            if dens > 0.3:
                for i in range(steps):
                    if i % 4 == 0:
                        continue
                    if np.random.random() < (dens - 0.3) * 0.4:
                        pat[i] = True
        elif track_type in ("snare", "clap"):
            for i in range(4, steps, 8):
                pat[i] = True
            if dens > 0.5:
                for i in range(steps):
                    if not pat[i] and np.random.random() < (dens - 0.4) * 0.5:
                        pat[i] = True
        elif "hat" in track_type:
            step_jump = 4 if dens < 0.5 else 2
            for i in range(0, steps, step_jump):
                if np.random.random() < dens + 0.2:
                    pat[i] = True
            if dens > 0.7 and step_jump == 2:
                for i in range(steps):
                    if not pat[i] and np.random.random() < (dens - 0.7) * 0.6:
                        pat[i] = True
        elif track_type in ("perc a", "perc b", "wood"):
            for i in range(steps):
                if np.random.random() < dens * 0.5:
                    pat[i] = True
        elif track_type == "reseq":
            for i in range(steps):
                if np.random.random() < dens:
                    pat[i] = True
        return pat


# ====================== UI WIDGETS ======================

class HoverFader:
    def __init__(self, owner, si=0.25, so=0.1):
        self.val = 0.0
        self.owner = owner; self.si = si; self.so = so; self.h = False

    def update(self):
        t = 1.0 if self.h else 0.0
        if abs(self.val - t) > 0.01:
            s = self.si if t > self.val else self.so
            self.val += (t - self.val) * s
            self.owner.update()
            return True
        elif self.val != t:
            self.val = t; self.owner.update()
        return False

    def enter(self): self.h = True; self.owner.update()
    def leave(self): self.h = False; self.owner.update()


class FadeButton(QPushButton):
    def __init__(self, text, parent=None, is_small=False, tooltip_text=""):
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.is_small = is_small
        self.hover = HoverFader(self, 0.2, 0.1)
        self.timer = QTimer(self); self.timer.setInterval(16)
        self.timer.timeout.connect(self._ua)
        if is_small:
            self.setMinimumSize(30, 18)
            self.bf = QFont("Segoe UI", 7, QFont.Weight.DemiBold)
        else:
            self.setMinimumSize(90, 26)
            self.bf = QFont("Segoe UI", 9, QFont.Weight.DemiBold)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        # Tooltip properties
        self.tooltip_text = tooltip_text
        self.tt_lbl = None
        self.tt_gfx = None
        self.tt_timer = QTimer(self); self.tt_timer.setInterval(16)
        self.tt_timer.timeout.connect(self._anim_tt)
        
        if tooltip_text:
            self.tt_lbl = QLabel(tooltip_text, self)
            self.tt_lbl.setStyleSheet("background: #2d3748; color: #ffffff; padding: 4px 8px; border-radius: 4px; font-size: 10px; font-family: 'Segoe UI';")
            self.tt_lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            self.tt_lbl.setVisible(False)
            self.tt_gfx = QGraphicsOpacityEffect(self.tt_lbl)
            self.tt_lbl.setGraphicsEffect(self.tt_gfx)
            self.tt_gfx.setOpacity(0.0)

    def _anim_tt(self):
        if self.tt_gfx is None: return
        op = self.tt_gfx.opacity()
        target = 1.0 if self.hover.h else 0.0
        if op == target:
            self.tt_timer.stop()
            if target == 0.0: self.tt_lbl.setVisible(False)
            return
        
        if self.hover.h:
            op = min(1.0, op + 0.15)
        else:
            op = max(0.0, op - 0.15)
        
        self.tt_gfx.setOpacity(op)
        
        if op > 0.0 and not self.tt_lbl.isVisible():
            self.tt_lbl.setVisible(True)
            self.tt_lbl.adjustSize()
            self.tt_lbl.move((self.width() - self.tt_lbl.width()) // 2, self.height() + 4)

    def _ua(self):
        if self.hover.update():
            self.update()
        else:
            self.timer.stop()

    def enterEvent(self, e): 
        self.hover.enter(); self.timer.start()
        if self.tt_lbl: self.tt_timer.start()

    def leaveEvent(self, e): 
        self.hover.leave(); self.timer.start()
        if self.tt_lbl: self.tt_timer.start()

    def paintEvent(self, ev):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        t = self.hover.val; r = self.rect().adjusted(1, 1, -1, -1)
        bg = QColor(255, 255, 255)
        if t > 0.01:
            bg = QColor(int(255 + (235 - 255) * t), int(255 + (248 - 255) * t), 255)
        bd = QColor(int(203 + (144 - 203) * t), int(213 + (205 - 213) * t), int(224 + (244 - 224) * t))
        tc = QColor(int(160 + (49 - 160) * t), int(174 + (130 - 174) * t), int(192 + (206 - 192) * t))
        p.setBrush(bg); p.setPen(QPen(bd, 1))
        p.drawRoundedRect(r, 13 if not self.is_small else 3, 13 if not self.is_small else 3)
        p.setPen(tc); p.setFont(self.bf)
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, self.text())


class SmoothKnob(QWidget):
    valueChanged = pyqtSignal(int)

    def __init__(self, base_hue=210, default_value=50, parent=None, is_bars=False, is_bpm=False, parent_block=None):
        super().__init__(parent)
        self.setMinimumSize(32, 32)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.SizeVerCursor)
        self.minimum = 0
        self.maximum = 100
        self.default_value = default_value
        self._value = default_value
        self.base_hue = base_hue
        self.is_bars = is_bars
        self.is_bpm = is_bpm
        self._parent_block = parent_block

        self.color_1 = QColor.fromHsl(int(self.base_hue), 150, 160)
        self.color_2 = QColor.fromHsl(int((self.base_hue + 20) % 360), 180, 130)

        self.fade_val = 0.0
        self.fade_timer = QTimer(self)
        self.fade_timer.setInterval(16)
        self.fade_timer.timeout.connect(self._animate_fade)
        self.show_value_timer = QTimer(self)
        self.show_value_timer.setInterval(600)
        self.show_value_timer.setSingleShot(True)
        self.show_value_timer.timeout.connect(self._start_fade_out)
        self._fading_out = False
        self.start_y = 0
        self.start_val = 0

    def _knob_size(self):
        h = self.height()
        w = self.width()
        if h <= 0 or w <= 0:
            return 32
        return max(24, min(h, w, 54)) # Increased from 50 to 54 to fill empty vertical space better

    def value(self):
        return self._value

    def setValue(self, val):
        val = max(self.minimum, min(self.maximum, val))
        if self._value != val:
            self._value = val
            self.valueChanged.emit(self._value)
            self._trigger_display()
            self.update()

    def _trigger_display(self):
        self._fading_out = False
        self.fade_val = 1.0
        self.show_value_timer.start()
        if not self.fade_timer.isActive():
            self.fade_timer.start()

    def _start_fade_out(self):
        self._fading_out = True

    def _animate_fade(self):
        if self._fading_out:
            self.fade_val -= 0.06
            if self.fade_val <= 0:
                self.fade_val = 0
                self.fade_timer.stop()
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.RightButton:
            self.setValue(self.default_value)
            e.accept()
        elif e.button() == Qt.MouseButton.LeftButton:
            self.start_y = e.position().y()
            self.start_val = self._value
            e.accept()

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.MouseButton.LeftButton:
            dy = self.start_y - e.position().y()
            self.setValue(int(self.start_val + dy * 1.2))
            e.accept()

    def wheelEvent(self, e):
        delta = e.angleDelta().y()
        if delta > 0:
            # Adjust the scroll step size as preferred
            self.setValue(self._value + 5)
        elif delta < 0:
            self.setValue(self._value - 5)  
        e.accept()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        size = self._knob_size()
        cx_k = (self.width() - size) / 2
        cy_k = (self.height() - size) / 2
        rect = QRectF(cx_k + 3, cy_k + 3, size - 6, size - 6)
        
        p.setPen(QPen(QColor("#e2e8f0"), 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawArc(rect, 225 * 16, -270 * 16)
        
        norm = (self._value - self.minimum) / max(1, self.maximum - self.minimum)
        
        if norm > 0:
            grad = QLinearGradient(rect.topLeft(), rect.bottomRight())
            grad.setColorAt(0, self.color_1)
            grad.setColorAt(1, self.color_2)
            p.setPen(QPen(grad, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawArc(rect, 225 * 16, int(-270 * norm * 16))
            
        angle = 225 - 270 * norm
        rad = math.radians(angle)
        cx = cx_k + size / 2
        cy = cy_k + size / 2
        r1 = size / 2 - 5
        r2 = size / 2 - 2
        x1, y1 = cx + r1 * math.cos(rad), cy - r1 * math.sin(rad)
        x2, y2 = cx + r2 * math.cos(rad), cy - r2 * math.sin(rad)
        p.setPen(QPen(self.color_1, 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(x1, y1), QPointF(x2, y2))
        
        if self.fade_val > 0:
            p.setPen(Qt.PenStyle.NoPen)
            bg_color = QColor(255, 255, 255, int(230 * self.fade_val))
            p.setBrush(bg_color)
            p.drawEllipse(rect.adjusted(2, 2, -2, -2))
            
            tc = QColor(int(self.color_1.red() * 0.7), int(self.color_1.green() * 0.7), int(self.color_1.blue() * 0.7), int(255 * self.fade_val))
            p.setPen(tc)
            fsize = max(7, int(size * 0.28))
            f = QFont("Segoe UI", fsize, QFont.Weight.DemiBold)
            p.setFont(f)

            if self.is_bars:
                val_str = f"{1 + int((self._value / 100.0) * 3.99)}b"
            elif self.is_bpm:
                val_str = str(int(60 + (self._value / 100.0) * 140))
            else:
                val_str = str(self._value)

            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, val_str)


class DropLabel(QLabel):
    file_dropped = pyqtSignal(str)
    clicked_signal = pyqtSignal()

    def __init__(self, text, default_style, loaded_style, hover_style):
        super().__init__(text)
        self.default_style = default_style
        self.loaded_style = loaded_style
        self.hover_style = hover_style
        self.is_loaded = False
        self.setStyleSheet(self.default_style)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.setMinimumWidth(40)

    def set_loaded(self, state):
        self.is_loaded = state
        self.setStyleSheet(self.loaded_style if state else self.default_style)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked_signal.emit()
            e.accept()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            for u in e.mimeData().urls():
                if u.isLocalFile() and u.toLocalFile().lower().endswith(('.wav', '.mp3', '.flac')):
                    e.acceptProposedAction()
                    self.setStyleSheet(self.hover_style)
                    return

    def dragLeaveEvent(self, e):
        self.setStyleSheet(self.loaded_style if self.is_loaded else self.default_style)

    def dropEvent(self, e):
        self.setStyleSheet(self.loaded_style if self.is_loaded else self.default_style)
        for u in e.mimeData().urls():
            if u.isLocalFile():
                fp = u.toLocalFile()
                if fp.lower().endswith(('.wav', '.mp3', '.flac')):
                    self.file_dropped.emit(fp)
                    break


class RndToggle(QWidget):
    toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(22, 16)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.active = False
        self.hover = HoverFader(self, 0.2, 0.1)
        self.timer = QTimer(self); self.timer.setInterval(16)
        self.timer.timeout.connect(self._ua)
        self.phase = 0.0

    def set_active(self, state):
        self.active = state
        self.toggled.emit(self.active)
        self.update()

    def _ua(self):
        if self.hover.update():
            self.update()
        else:
            self.timer.stop()

    def enterEvent(self, e): self.hover.enter(); self.timer.start()
    def leaveEvent(self, e): self.hover.leave(); self.timer.start()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.active = not self.active
            self.toggled.emit(self.active); self.update()

    def paintEvent(self, ev):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        t = self.hover.val; r = self.rect().adjusted(1, 1, -1, -1)
        if self.active:
            bg = QColor(236, 243, 247); bd = QColor(159, 192, 214); tc = QColor(88, 126, 164)
        else:
            bg = QColor(int(255 + (235 - 255) * t), int(255 + (248 - 255) * t), 255)
            bd = QColor(int(203 + (144 - 203) * t), int(213 + (205 - 213) * t), int(224 + (244 - 224) * t))
            tc = QColor(int(160 + (49 - 160) * t), int(174 + (130 - 174) * t), int(192 + (206 - 192) * t))
        p.setBrush(bg); p.setPen(QPen(bd, 1))
        p.drawRoundedRect(r, 3, 3)
        p.setPen(tc); p.setFont(QFont("Segoe UI", 7, QFont.Weight.DemiBold))
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, "r")
        if self.active:
            self.phase += 0.15
            a = int(60 + np.sin(self.phase) * 40)
            c = QColor(72, 163, 236, a)
            g = QRadialGradient(float(r.center().x()), float(r.center().y()), float(r.width() * 0.7))
            g.setColorAt(0, c); g.setColorAt(1, QColor(255, 255, 255, 0))
            p.setBrush(g); p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(r, 3, 3)


class StatusWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(22)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.text = ""; self.opacity = 0.0; self.target = 0.0; self.active = False
        self.at = QTimer(self); self.at.timeout.connect(self._anim); self.at.setInterval(16)
        self.dt = QTimer(self); self.dt.setSingleShot(True)
        self.dt.timeout.connect(self._fo)
        self.font = QFont("Segoe UI", 9, QFont.Weight.DemiBold)

    def set_text(self, t):
        self.dt.stop(); self.text = t; self.target = 1.0; self.active = True
        if not self.at.isActive(): self.at.start()
        self.dt.start(3000); self.update()

    def _fo(self): self.target = 0.0

    def _anim(self):
        s = 0.1
        if self.opacity < self.target:
            self.opacity = min(self.target, self.opacity + s)
        elif self.opacity > self.target:
            self.opacity = max(self.target, self.opacity - s)
        if self.opacity <= 0.01 and self.target == 0:
            self.opacity = 0; self.active = False; self.at.stop(); self.update(); return
        if self.active: self.update()

    def paintEvent(self, ev):
        if self.opacity <= 0: return
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = QColor("#718096"); c.setAlpha(int(255 * self.opacity))
        p.setPen(c); p.setFont(self.font)
        p.drawText(self.rect().adjusted(8, 0, -5, 0),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.text)


class WaveformPlayer(QWidget):
    seek_requested = pyqtSignal(float)
    import_clicked = pyqtSignal()
    scrub_started = pyqtSignal()
    scrub_ended = pyqtSignal()
    file_dropped = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.data = None
        self.pos = 0.0
        self.setMinimumHeight(60)
        self.setMaximumHeight(80)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._scrub = False
        self.phase = 0.0

        self._path = None
        self.old_path = None
        self.wf_opacity = 1.0
        self.old_opacity = 0.0

        self._rt = QTimer(self); self._rt.setSingleShot(True)
        self._rt.timeout.connect(lambda: self._upd(fade=False))
        self.tt = QTimer(self); self.tt.timeout.connect(self._anim); self.tt.start(40)

    def _anim(self):
        self.phase = (self.phase + 0.005) % 1.0
        
        if self.wf_opacity < 1.0:
            self.wf_opacity = min(1.0, self.wf_opacity + 0.45)
        if self.old_opacity > 0.0:
            self.old_opacity = max(0.0, self.old_opacity - 0.45)
            if self.old_opacity <= 0.0:
                self.old_path = None
                
        self.update()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            for u in e.mimeData().urls():
                if u.isLocalFile() and u.toLocalFile().lower().endswith(('.wav', '.mp3', '.flac')):
                    e.acceptProposedAction()
                    return

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            if u.isLocalFile():
                fp = u.toLocalFile()
                if fp.lower().endswith(('.wav', '.mp3', '.flac')):
                    self.file_dropped.emit(fp)
                    break

    def set_data(self, d):
        if d is None or len(d) == 0:
            self.data = None; self._path = None; self.update(); return
        if d.ndim > 1: dm = d.mean(axis=1)
        else: dm = d
        tp = 1500; s = max(1, len(dm) // tp)
        self.data = dm[::s]; self.pos = 0.0
        self._upd(fade=True)

    def set_pos(self, p):
        if not self._scrub:
            self.pos = max(0, min(1, p)); self.update()

    def resizeEvent(self, e):
        self._rt.start(50); super().resizeEvent(e)

    def mousePressEvent(self, e):
        if self.data is None:
            self.import_clicked.emit()
        elif e.button() == Qt.MouseButton.LeftButton:
            self._scrub = True; self.scrub_started.emit(); self._hi(e.pos().x())

    def mouseMoveEvent(self, e):
        if self.data is not None and self._scrub:
            w = self.width()
            if w > 0:
                self.pos = max(0, min(w, e.pos().x())) / w; self.update()

    def mouseReleaseEvent(self, e):
        if self._scrub:
            self._scrub = False
            self.seek_requested.emit(self.pos); self.scrub_ended.emit()

    def _hi(self, x):
        w = self.width()
        if w > 0:
            self.pos = max(0, min(w, x)) / w
            self.seek_requested.emit(self.pos); self.update()

    def _upd(self, fade=False):
        w, h = self.width(), self.height()
        if w == 0 or h == 0 or self.data is None:
            self._path = None; self.update(); return

        path = QPainterPath()
        cy = h / 2
        path.moveTo(0, cy)
        xs = w / len(self.data); asc = h * 0.45
        for i, v in enumerate(self.data):
            path.lineTo(i * xs, cy - (v * asc))
        for i in range(len(self.data) - 1, -1, -1):
            path.lineTo(i * xs, cy + (self.data[i] * asc))
        path.closeSubpath()

        if fade:
            self.old_path = getattr(self, '_path', None)
            self.old_opacity = getattr(self, 'wf_opacity', 1.0)
            self.wf_opacity = 0.0
        else:
            self.old_path = None
            self.old_opacity = 0.0
            self.wf_opacity = 1.0
            
        self._path = path
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect(); w, h = r.width(), r.height()
        p.fillRect(r, QColor(247, 250, 252, 200))

        if self.data is None and self._path is None:
            g = QLinearGradient(0, 0, r.width(), 0)
            for i in range(4):
                t = i / 3; hv = (self.phase + t * 0.3) % 1.0
                g.setColorAt(t, QColor.fromHslF(hv, 0.6, 0.65, 1.0))
            p.setFont(QFont("Segoe UI", 10))
            p.setPen(QPen(QBrush(g), 0))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "click generate or drop audio")
            return

        def draw_rainbow_path(path, opacity):
            if path is None or opacity <= 0: return
            p.setOpacity(opacity)
            g = QLinearGradient(0, 0, w, 0)
            for i in range(6):
                t = i / 5.0
                hv = (self.phase + t * 1.5) % 1.0
                g.setColorAt(t, QColor.fromHslF(hv, 0.69, 0.69, 0.9))
            p.setBrush(g)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPath(path)

        draw_rainbow_path(self.old_path, self.old_opacity)
        draw_rainbow_path(self._path, self.wf_opacity)

        p.setOpacity(1.0)
        if self.pos >= 0 and self._path is not None:
            px = int(self.pos * w)
            lg = QLinearGradient(px, 0, px, h)
            for i in range(5):
                t = i / 4; hv = (t * 0.25 + self.pos * 2.5) % 1.0
                lg.setColorAt(t, QColor.fromHslF(hv, 0.69, 0.69, 0.9))
            p.setBrush(lg); p.setPen(Qt.PenStyle.NoPen)
            p.drawRect(QRectF(px - 1, 0, 2, h))


class PlayButton(QWidget):
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(70, 26)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._state = 0; self.av = 0.0; self.im = 0.0
        self.hover = HoverFader(self, 0.2, 0.1)
        self.t = QTimer(self); self.t.timeout.connect(self._a); self.t.start(20)

    def set_playing(self, s): self._state = 1 if s else 0

    def mousePressEvent(self, e): self.clicked.emit()
    def enterEvent(self, e): self.hover.enter()
    def leaveEvent(self, e): self.hover.leave()

    def _a(self):
        ta = 0.0 if self._state == 0 else 1.0
        if abs(self.av - ta) > 0.001: self.av += (ta - self.av) * 0.25
        tm = 1.0 if self._state == 1 else 0.0
        if abs(self.im - tm) > 0.001: self.im += (tm - self.im) * 0.25
        self.hover.update(); self.update()

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect(); c = QRectF(r).center(); t = self.hover.val
        p.setBrush(QColor(int(226 + (203 - 226) * t), int(232 + (213 - 232) * t), int(240 + (224 - 240) * t)))
        p.setPen(Qt.PenStyle.NoPen); p.drawRoundedRect(r, 13, 13)
        fg = QColor("#4a5568"); p.translate(c)
        sa = 0.8 + 0.2 * self.av; p.scale(sa, sa); m = self.im
        path = QPainterPath()
        p1x = -3 * (1 - m) + (-5) * m
        p2x = -3 * (1 - m) + (-5) * m
        p3x = 6 * (1 - m) + (-2) * m; p3y = 0 * (1 - m) + 5 * m
        p4x = 6 * (1 - m) + (-2) * m; p4y = 0 * (1 - m) + (-5) * m
        path.moveTo(p1x, -5); path.lineTo(p2x, 5); path.lineTo(p3x, p3y); path.lineTo(p4x, p4y); path.closeSubpath()
        p.setBrush(fg); p.drawPath(path)
        if m > 0.01:
            fg.setAlpha(int(255 * m)); p.setBrush(fg)
            off = (1 - m) * 2
            p.drawRect(QRectF(2 + off, -5, 3, 10))


# ====================== COMPACT PARAM COLUMN ======================

class ParamCol(QWidget):
    def __init__(self, name, hue, default=50, is_bars=False, is_bpm=False, show_rnd=True): # <-- Add show_rnd=True
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(1)

        lbl = QLabel(name.lower())
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setFixedHeight(13)
        lbl.setStyleSheet("font-size:9px;font-weight:600;color:#718096;font-family:'Segoe UI';")
        lay.addWidget(lbl)

        self.rnd = None
        if show_rnd: # <-- Wrap the RndToggle instantiation
            self.rnd = RndToggle()
            wrap = QWidget(); wl = QHBoxLayout(wrap)
            wrap.setFixedHeight(16)
            wl.setContentsMargins(0, 0, 0, 0); wl.addWidget(self.rnd, 0, Qt.AlignmentFlag.AlignHCenter)
            lay.addWidget(wrap)

        self.knob = SmoothKnob(base_hue=hue, default_value=default, is_bars=is_bars, is_bpm=is_bpm, parent_block=self)
        lay.addWidget(self.knob, 1, Qt.AlignmentFlag.AlignHCenter)

    def value(self):
        return self.knob.value() / 100.0


# ====================== COMPACT DRUM TRACK BLOCK ======================

class DrumTrackBlock(QFrame):
    def __init__(self, label, dtype, hue):
        super().__init__()
        self.label_text = label
        self.dtype = dtype
        self.base_hue = hue
        self.raw_sample = None
        self.is_sample = False
        self.current_data = None
        self.original_data = None
        self.sp = {'pitch': 0.5, 'decay': 0.5, 'tone': 0.3}
        self.kit_idx = np.random.randint(0, 4)

        self.setStyleSheet("DrumTrackBlock{background:#f7fafc;border:1px solid #e2e8f0;border-radius:6px;}")
        self.setMinimumHeight(70)
        self.setMaximumHeight(95)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        main = QHBoxLayout(self)
        main.setContentsMargins(4, 2, 4, 2); main.setSpacing(6)

        ds = "color:#4a5568;font-weight:600;font-size:11px;font-family:'Segoe UI';"
        ls = "color:#38a169;font-weight:600;font-size:11px;font-family:'Segoe UI';"
        hs = "color:#ffffff;background:#48bb78;font-weight:600;font-size:11px;font-family:'Segoe UI';border-radius:4px;"
        
        self.lbl = DropLabel(label.lower(), ds, ls, hs)
        self.lbl.clicked_signal.connect(self._open_file)
        self.lbl.file_dropped.connect(self.load_sample)
        main.addWidget(self.lbl)

        self.params = {}
        cfg = [("dens", 0, 40), ("vol", 10, 85), ("pitch", 20, 50),
               ("decay", 30, 50), ("tone", 40, 30), ("crush", 50, 0), ("filt", 60, 50)]
        
        for name, hoff, dv in cfg:
            col = ParamCol(name, (hue + hoff) % 360, dv)
            main.addWidget(col)
            self.params[name] = col

        self.ut = QTimer(self); self.ut.setSingleShot(True); self.ut.setInterval(120)
        self.ut.timeout.connect(lambda: self.update_sound())
        self.update_sound()

    def _open_file(self):
        fn, _ = QFileDialog.getOpenFileName(self, "load sample", "", "Audio (*.wav *.mp3 *.flac)")
        if fn:
            self.load_sample(fn)

    def load_sample(self, fp):
        try:
            d, fs = sf.read(fp)
            if d.ndim > 1: d = d.mean(axis=1)
            if len(d) > 3 * fs: d = d[:3 * fs]
            if fs != SR: d = signal.resample(d, int(len(d) * SR / fs))
            pk = np.max(np.abs(d))
            if pk > 0: d = d / pk
            self.raw_sample = SynthEngine.ensure_zero_crossing(d.astype(np.float32))
            self.is_sample = True
            self.lbl.set_loaded(True)
            self.update_sound()
        except Exception:
            pass

    def schedule(self):
        self.ut.start()

    def get_vals(self):
        return {k: v.value() for k, v in self.params.items()}

    def randomize_active(self):
        for k, col in self.params.items():
            if col.rnd and col.rnd.active: # Added col.rnd safety check
                if k == "vol": col.knob.setValue(np.random.randint(70, 95))
                elif k == "crush": col.knob.setValue(np.random.randint(0, 40))
                elif k == "filt": col.knob.setValue(np.random.randint(20, 80))
                elif k == "dens": col.knob.setValue(np.random.randint(20, 75))
                else: col.knob.setValue(np.random.randint(20, 80))

    def update_sound(self):
        p = self.params['pitch'].value()
        d = self.params['decay'].value()
        t = self.params['tone'].value()
        self.sp = {'pitch': p, 'decay': d, 'tone': t}
        if self.is_sample and self.raw_sample is not None:
            self.original_data = SynthEngine.process_sample(self.raw_sample, self.sp)
        else:
            # Temporarily apply this track's specific kit index
            old_kit = SynthEngine.current_kit
            SynthEngine.current_kit = getattr(self, 'kit_idx', 2)
            self.original_data = SynthEngine.generate_drum(self.dtype, self.sp)
            SynthEngine.current_kit = old_kit
        self._process()

    def _process(self):
        if self.original_data is None or len(self.original_data) == 0:
            self.original_data = np.zeros(1024, dtype=np.float32)
        f = SynthEngine.apply_filter(self.original_data, self.params['filt'].value())
        self.current_data = SynthEngine.resample_lofi(f, self.params['crush'].value())
        v = self.params['vol'].value() ** 2
        self.current_data *= v
        if len(self.current_data) > 100:
            self.current_data[-50:] *= np.linspace(1, 0, 50)
        self.current_data = np.nan_to_num(self.current_data, copy=False)
        if self.current_data is None:
            self.current_data = np.zeros(1024, dtype=np.float32)

    def build_slot(self, steps=16):
        dens = self.params['dens'].value()
        pat = PatternGen.build(self.dtype, dens, steps)
        vels = [np.random.uniform(0.5, 1.0) if active else 0.8 for active in pat]
        return {
            'data': self.current_data,
            'pattern': pat,
            'velocities': vels,
            'is_sliced': False,
            'is_bass': False,
            'label': self.label_text.lower()
        }


# ====================== COMPACT RESEQ BLOCK ======================

class ReseqBlock(QFrame):
    def __init__(self, bpm_ref, hue=260):
        super().__init__()
        self.label_text = "reseq"
        self.bpm = bpm_ref
        self.raw_sample = None
        self.current_data = None

        self.setStyleSheet("ReseqBlock{background:#f7fafc;border:1px solid #90cdf4;border-radius:6px;}")
        self.setMinimumHeight(70)
        self.setMaximumHeight(95)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        main = QHBoxLayout(self)
        main.setContentsMargins(4, 2, 4, 2); main.setSpacing(6)

        ds = "color:#2b6cb0;font-weight:600;font-size:11px;font-family:'Segoe UI';border:0px solid #90cdf4;border-radius:4px;"
        ls = "color:#2b6cb0;font-weight:600;font-size:10px;font-family:'Segoe UI';border:1px solid #90cdf4;border-radius:4px;background:#ffffff;"
        hs = "color:#ffffff;background:#4299e1;font-weight:600;font-size:11px;font-family:'Segoe UI';border-radius:4px;"
        
        self.lbl = DropLabel("reseq", ds, ls, hs)
        self.lbl.clicked_signal.connect(self._open)
        self.lbl.file_dropped.connect(self.load)
        main.addWidget(self.lbl)

        self.params = {}
        cfg = [("dens", 0, 40), ("vol", 10, 80), ("pitch", 20, 50),
               ("decay", 30, 50), ("tone", 40, 50), ("crush", 50, 0), ("filt", 60, 50)]
        for name, hoff, dv in cfg:
            # Pass show_rnd=False here
            col = ParamCol(name, (hue + hoff) % 360, dv, show_rnd=False)
            main.addWidget(col)
            self.params[name] = col

        self.ut = QTimer(self); self.ut.setSingleShot(True); self.ut.setInterval(80)
        self.ut.timeout.connect(self._process)
        self._process()

    def _open(self):
        fn, _ = QFileDialog.getOpenFileName(self, "load reseq sample", "", "Audio (*.wav *.mp3 *.flac)")
        if fn:
            self.load(fn)

    def load(self, fp):
        try:
            d, fs = sf.read(fp)
            if d.ndim > 1: d = d.mean(axis=1)
            if fs != SR: d = signal.resample(d, int(len(d) * SR / fs))
            pk = np.max(np.abs(d))
            if pk > 0: d = d / pk
            self.raw_sample = np.nan_to_num(d.astype(np.float32))
            
            short_name = os.path.basename(fp)[:8].lower()
            self.lbl.setText(short_name)
            self.lbl.set_loaded(True)
            self._process()
        except Exception:
            pass

    def schedule(self): self.ut.start()

    def get_vals(self):
        return {k: v.value() for k, v in self.params.items()}

    def randomize_active(self):
        for k, col in self.params.items():
            if col.rnd and col.rnd.active: # Added col.rnd safety check
                if k == "vol": col.knob.setValue(np.random.randint(65, 90))
                elif k == "crush": col.knob.setValue(np.random.randint(0, 50))
                elif k == "dens": col.knob.setValue(np.random.randint(25, 70))
                else: col.knob.setValue(np.random.randint(20, 80))

    def _process(self, steps=16):
        params = {
            'pitch': self.params['pitch'].value(),
            'tone': self.params['tone'].value(),
            'filter': self.params['filt'].value(),
            'crush': self.params['crush'].value(),
            'decay': self.params['decay'].value()
        }
        if self.raw_sample is not None:
            r = ReseqEngine.process(self.raw_sample, self.bpm, params, steps)
        else:
            r = np.zeros(int((60 / self.bpm) * 4 * (steps // 16) * SR), dtype=np.float32)
        v = self.params['vol'].value() ** 2
        self.current_data = np.nan_to_num(r * v, copy=False)
        return self.current_data

    def update_bpm(self, b):
        self.bpm = b; self._process()

    def build_slot(self, steps=16):
        data = self._process(steps)
        dens = self.params['dens'].value()
        pat = PatternGen.build("reseq", dens, steps)
        vels = [np.random.uniform(0.5, 1.0) if active else 0.8 for active in pat]
        return {
            'data': data,
            'pattern': pat,
            'velocities': vels,
            'is_sliced': True,
            'is_bass': False,
            'label': 'reseq'
        }


# ====================== COMPACT MASTER FX ROW ======================

class MasterFxRow(QFrame):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("MasterFxRow{background:#f7fafc;border:1px solid #cbd5e0;border-radius:6px;}")
        self.setMinimumHeight(70)
        self.setMaximumHeight(95)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        main = QHBoxLayout(self)
        main.setContentsMargins(8, 2, 8, 2); main.setSpacing(10)

        lbl = QLabel("master")
        lbl.setFixedWidth(40)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color:#2b6cb0;font-weight:600;font-size:11px;font-family:'Segoe UI';")
        main.addWidget(lbl)

        self.params = {}
        cfg = [
            ("bpm",   "bpm",       200, 50, "bpm"),
            ("swng",  "swing",     220, 0,  "lin"),
            ("cut",   "clip",      240, 0,  "lin"),
            ("rate",  "rate",      260, 50, "rate"),
            ("v/pan", "vpan",      280, 0,  "lin"),
            ("rvb",   "reverb",    300, 10, "lin"),
            ("bit",   "crush",     320, 0,  "lin"),
            ("flt",   "filter_amt", 340, 0, "lin"),
            ("tone",  "tone",      160, 50, "tone"),
            ("len",   "length",    180, 50, "bars"),
        ]
        for name, key, hue, dv, mapping in cfg:
            # Pass show_rnd=False here
            col = ParamCol(name, hue, dv, is_bars=(mapping=="bars"), is_bpm=(mapping=="bpm"), show_rnd=False)
            main.addWidget(col)
            self.params[key] = (col, mapping)

    def randomize_active(self):
        for key, (col, mapping) in self.params.items():
            if col.rnd and col.rnd.active: # Added col.rnd safety check
                if key == "bpm": col.knob.setValue(np.random.randint(35, 75))
                elif key == "rate": col.knob.setValue(np.random.randint(40, 60))
                elif key == "reverb": col.knob.setValue(np.random.randint(0, 40))
                elif key == "crush": col.knob.setValue(np.random.randint(0, 40))
                elif key == "clip": col.knob.setValue(np.random.randint(0, 30))
                elif key == "tone": col.knob.setValue(np.random.randint(30, 70))
                elif key == "length": pass 
                else: col.knob.setValue(np.random.randint(0, 60))

    def get_vals(self):
        out = {}
        for key, (col, mapping) in self.params.items():
            v = col.value()
            if mapping == "bpm":
                out['bpm'] = int(60 + v * 140)
            elif mapping == "rate":
                out['rate'] = 0.5 + v
            elif mapping == "tone":
                out['tone'] = v
            elif mapping == "bars":
                out['length'] = 1 + int(v * 3.99)
            else:
                out[key] = v
        return out


# ====================== GENERATE THREAD ======================

class GenerateThread(QThread):
    finished_ok = pyqtSignal(object, int, float)
    error = pyqtSignal(str)

    def __init__(self, slots, master, fx):
        super().__init__()
        self.slots = slots
        self.master = master
        self.fx = fx

    def run(self):
        t0 = time.perf_counter()
        try:
            bpm = self.master.get('bpm', 120)
            swing = self.master.get('swing', 0)
            clip = self.master.get('clip', 0)
            vpan = self.master.get('vpan', 0)
            rate = self.master.get('rate', 1.0)
            reverb = self.master.get('reverb', 0)
            crush = self.master.get('crush', 0)
            tone = self.master.get('tone', 0.5)
            filter_amt = self.master.get('filter_amt', 0)
            bars = self.master.get('length', 1)
            
            steps = bars * 16
            slot_data = [s.build_slot(steps) for s in self.slots]
            mix = AudioMixer.mix(slot_data, bpm, swing, clip, 0, steps)

            sr = SR
            y = mix
            if filter_amt > 0:
                y = Effects.apply_rand_filter(y, sr, filter_amt, bpm)
            if vpan > 0:
                y = Effects.apply_vol_pan(y, sr, vpan, bpm)
            if y.ndim == 1:
                y = np.column_stack((y, y))
            if crush > 0:
                y = Effects.bitcrush(y, sr, crush)
            if tone != 0.5:
                y = Effects.apply_tone(y, sr, tone)
            if rate != 1.0 and rate > 0:
                y = signal.resample(y, int(len(y) / rate))
            if reverb > 0:
                y = Effects.simple_reverb(y, sr, mix=reverb * 0.35, room=0.6, damp=0.6)
            peak = np.max(np.abs(y))
            if peak > 1:
                y = y / peak
                
            t1 = time.perf_counter()
            self.finished_ok.emit(y.astype(np.float32), sr, t1 - t0)
        except Exception as e:
            self.error.emit(str(e))


# ====================== MAIN WINDOW ======================

class AutomaWindow(QMainWindow):
    FIXED_WINDOW_SIZE = (936, 369)
    def __init__(self):
        super().__init__()
        self.setWindowTitle("automa")
        if self.FIXED_WINDOW_SIZE:
            self.setFixedSize(*self.FIXED_WINDOW_SIZE)
        else:
            self.setMinimumSize(936, 369)
        self.slots = []
        self.temp_file = None
        self.is_generating = False
        self.last_wall = 0
        self.last_pos = 0
        self.processed = None

        self.player = QMediaPlayer()
        self.ao = QAudioOutput()
        self.player.setAudioOutput(self.ao)
        self.player.mediaStatusChanged.connect(self._media_status)

        self.anim_t = QTimer(self); self.anim_t.setInterval(15)
        self.anim_t.timeout.connect(self._hf)

        self._setup_ui()

    def _setup_ui(self):
        cw = QWidget(); self.setCentralWidget(cw)
        cw.setStyleSheet("QWidget{font-family:'Segoe UI',sans-serif;background-color:#f7fafc;}")

        main = QVBoxLayout(cw)
        main.setContentsMargins(10, 8, 10, 10); main.setSpacing(4)

        # --- top row (master, reseq) ---
        row0_lay = QHBoxLayout()
        row0_lay.setSpacing(8)
        self.master_row = MasterFxRow()
        self.reseq = ReseqBlock(120, hue=260)
        row0_lay.addWidget(self.master_row, stretch=10)
        row0_lay.addWidget(self.reseq, stretch=7)
        main.addLayout(row0_lay)
        
        # --- middle row (kick, snare, hat c) ---
        row1_lay = QHBoxLayout()
        row1_lay.setSpacing(8)
        self.kick = DrumTrackBlock("kick", "kick", 0)
        self.snare = DrumTrackBlock("snare", "snare", 30)
        self.hat_c = DrumTrackBlock("hat c", "closed hat", 60)
        row1_lay.addWidget(self.kick)
        row1_lay.addWidget(self.snare)
        row1_lay.addWidget(self.hat_c)
        main.addLayout(row1_lay)

        # --- bottom row (hat o, clap, perc) ---
        row2_lay = QHBoxLayout()
        row2_lay.setSpacing(8)
        self.hat_o = DrumTrackBlock("hat o", "open hat", 90)
        self.clap = DrumTrackBlock("clap", "clap", 150)
        self.perc = DrumTrackBlock("perc", "perc a", 200)
        row2_lay.addWidget(self.hat_o)
        row2_lay.addWidget(self.clap)
        row2_lay.addWidget(self.perc)
        main.addLayout(row2_lay)

        self.slots.extend([self.kick, self.snare, self.hat_c, self.hat_o, self.clap, self.perc, self.reseq])

        for s in self.slots:
            for col in s.params.values():
                col.knob.valueChanged.connect(s.schedule)

        # --- buttons row ---
        btns = QHBoxLayout(); btns.setContentsMargins(0, 4, 0, 4); btns.setSpacing(6)
        
        # UTMOST LEFT: Play Button
        self.btn_play = PlayButton()
        self.btn_play.clicked.connect(self._toggle_play)
        btns.addWidget(self.btn_play)

        # SECOND: Generate Button
        self.btn_gen = FadeButton("generate", tooltip_text="click or press enter")
        self.btn_gen.clicked.connect(self.generate)
        btns.addWidget(self.btn_gen)

        # THIRD: Export Button
        self.btn_exp = FadeButton("export")
        self.btn_exp.clicked.connect(self._export)
        btns.addWidget(self.btn_exp)

        # FOURTH: Randomize Button
        self.btn_rnd_all = FadeButton("randomize")
        self.btn_rnd_all.clicked.connect(self._toggle_all_rnd)
        btns.addWidget(self.btn_rnd_all)

        btns.addStretch()

        main.addLayout(btns)

        # --- waveform player ---
        self.wave = WaveformPlayer()
        self.wave.seek_requested.connect(self._seek)
        self.wave.import_clicked.connect(self._open_file)
        self.wave.scrub_started.connect(self.anim_t.stop)
        self.wave.scrub_ended.connect(self._resume)
        self.wave.file_dropped.connect(self._wave_dropped)
        main.addWidget(self.wave)

        # --- status bar ---
        self.status = StatusWidget()
        main.addWidget(self.status)

        self._update_preview()

    # ---------- controls ----------

    def _toggle_all_rnd(self):
        state = not getattr(self, '_all_rnd_state', False)
        self._all_rnd_state = state
        # Limit randomization changes strictly to active drum tracks
        for s in self.slots:
            if isinstance(s, DrumTrackBlock):
                for col in s.params.values():
                    col.rnd.set_active(state)
        self._notify(f"drum randomize {'on' if state else 'off'}")

    def _notify(self, t):
        self.status.set_text(t)

    def _open_file(self):
        fn, _ = QFileDialog.getOpenFileName(self, "import audio", "", "Audio (*.wav *.mp3 *.flac)")
        if fn:
            self.reseq.load(fn)
            self._notify(f"reseq: {os.path.basename(fn).lower()}")

    def _wave_dropped(self, fp):
        self.reseq.load(fp)
        self._notify(f"reseq: {os.path.basename(fp).lower()}")

    def _update_preview(self):
        try:
            m = self.master_row.get_vals()
            bars = m.get('length', 1)
            steps = bars * 16
            slot_data = [s.build_slot(steps) for s in self.slots]
            mix = AudioMixer.mix(slot_data, m.get('bpm', 120), m.get('swing', 0),
                                 m.get('clip', 0), 0, steps)
            self.wave.set_data(mix)
        except Exception:
            pass

    # ---------- generate ----------

    def generate(self):
        if self.is_generating:
            return
        
        self.player.stop()
        self.player.setSource(QUrl())
        self.anim_t.stop()
        self.btn_play.set_playing(False)
        
        self.is_generating = True
        self._notify("generating...")

        # SynthEngine.set_kit(np.random.randint(0, 4)) - removed for more randomness

        for s in self.slots:
            if isinstance(s, DrumTrackBlock):
                s.kit_idx = np.random.randint(0, 4)
            s.randomize_active()
            s.schedule()

        m = self.master_row.get_vals()

        self.gt = GenerateThread(self.slots, m, m)
        self.gt.finished_ok.connect(self._gen_done)
        self.gt.error.connect(self._gen_err)
        self.gt.start()

    def _gen_done(self, data, sr, gen_time):
        self.processed = data
        self.wave.set_data(data)
        try:
            self.player.stop()
            self.player.setSource(QUrl())
            if self.temp_file and os.path.exists(self.temp_file):
                try: os.remove(self.temp_file)
                except: pass
            fd, tp = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            sf.write(tp, data, sr)
            self.temp_file = tp
            self.player.setSource(QUrl.fromLocalFile(tp))
            self.last_pos = 0
            self.last_wall = time.perf_counter()
            self.player.play()
            self.btn_play.set_playing(True)
            self.anim_t.start()
            self._notify(f"generated in {gen_time:.3f}s")
        except Exception as e:
            print(f"temp err: {e}")
        self.is_generating = False

    def _gen_err(self, m):
        self._notify(f"err: {m}")
        self.is_generating = False

    # ---------- playback ----------

    def _toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause(); self.anim_t.stop()
            self.btn_play.set_playing(False)
        else:
            self.last_wall = time.perf_counter()
            self.last_pos = self.player.position()
            self.player.play(); self.anim_t.start()
            self.btn_play.set_playing(True)

    def _resume(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.last_wall = time.perf_counter()
            self.last_pos = self.player.position()
            self.anim_t.start()

    def _hf(self):
        d = self.player.duration()
        if d > 0 and self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            now = time.perf_counter()
            delta = now - self.last_wall
            ip = self.last_pos + delta * 1000
            ap = self.player.position()
            if abs(ip - ap) > 150:
                self.last_pos = ap; self.last_wall = now; ip = ap
            self.wave.set_pos(ip / d)

    def _seek(self, p):
        d = self.player.duration()
        if d > 0:
            ms = int(p * d)
            self.player.setPosition(ms)
            self.last_pos = ms
            self.last_wall = time.perf_counter()

    def _media_status(self, s):
        if s == QMediaPlayer.MediaStatus.EndOfMedia:
            self.btn_play.set_playing(False)
            self.wave.set_pos(0)
            self.anim_t.stop()

    # ---------- export ----------

    def _export(self):
        if self.processed is None:
            self._notify("generate first")
            return
        home = os.path.expanduser("~")
        sd = os.path.join(home, "Music", "automa")
        if not os.path.exists(sd):
            try: os.makedirs(sd)
            except:
                self._notify("err: folder"); return
        ts = int(time.time())
        fp = os.path.join(sd, f"automa_{ts}.wav")
        try:
            sf.write(fp, self.processed, SR)
            self._notify("saved to: music/automa")
        except Exception as e:
            self._notify(f"err: {e}")

    def closeEvent(self, e):
        if self.temp_file and os.path.exists(self.temp_file):
            try: os.remove(self.temp_file)
            except: pass
        e.accept()

    def keyPressEvent(self, event):
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.generate()
            else:
                super().keyPressEvent(event)

# ====================== ENTRY ======================

if __name__ == '__main__':
    import os
    os.environ["QT_SCALE_FACTOR"] = "1.05"
    
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('automa.audio.tool.v3')
    except:
        pass

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(sys.argv)
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "automa.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    elif os.path.exists("automa.ico"):
        app.setWindowIcon(QIcon("automa.ico"))

    w = AutomaWindow()
    w.show()
    sys.exit(app.exec())