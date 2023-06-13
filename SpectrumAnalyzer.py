"""
    For streaming data from a microphone in realtime

    Audio data is attained using pyaudio
    then converted from binary data to ints using struct
    then displayed using matplotlib

    Top figure plots audio data as an Oscilloscope (Time vs. Intensity)
    Bottom figure plots audio data as a Spectrum Analyzer (Frequency  vs. Intensity)

    Green dot is a marker which follows the frequency the mouse pointer is on.

    Press any keyboard key to enable single shot....
    Single shot freezes trace but still allows for marker movement...
    press any keyboard key again to restart SpecAn
"""


import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from matplotlib.animation import FuncAnimation
import struct


class AudioStream:


    def __init__(self):
        self.RATE = 44100
        self.CHANNEL = 1
        self.CHUNK = 1024*8
        self.FORMAT = pyaudio.paInt16
        self.Xmark = 0
        self.Ymark = 0
        self.index = 0
        self.FREQ = np.linspace(0, int(self.RATE/2), int(self.CHUNK/2))
        self.xdata = np.linspace(0, self.CHUNK/self.RATE, self.CHUNK)
        p = pyaudio.PyAudio()
        self.stream = p.open(rate=self.RATE, channels=self.CHANNEL, format=self.FORMAT, input=True, output=True, frames_per_buffer=self.CHUNK)
    
    def create_graph(self):
        self.fig, (self.ax, self.ax2) = plt.subplots(2)
        self.graph_style()
        self.Oscope, = self.ax.plot(self.xdata, np.sin(self.xdata))
        self.SpecAn, = self.ax2.plot(self.FREQ, self.FREQ)
        self.Peak, = self.ax2.plot(0, 0, marker='o')
        self.Marker, = self.ax2.plot(0,0, marker='o')

    def graph_style(self):
        self.ax.set_ylim(-127, 300)
        self.ax.set_xlim(0, self.CHUNK/self.RATE)
        self.ax2.set_ylim(0, 1)
        self.ax2.set_xlim(50, 10000)
        self.ax.title.set_text("Oscilloscope")
        self.ax.set_ylabel("Level")
        self.ax.set_xlabel("Time in seconds")
        self.ax2.title.set_text("Spectrum Analyzer")
        self.ax2.set_ylabel("Level")
        self.ax2.set_xlabel("Frequency Hz")
        self.text = self.ax2.text(500, 0.5, f"Peak: {0}Hz, {0}V")
        self.text2 = self.ax2.text(8000, 0.9, f"Marker: {0}Hz, {0}V")


    def raw_data(self):
        data = self.stream.read(self.CHUNK)
        data_int = struct.unpack(str(2 * self.CHUNK) + 'B', data)
        return data_int

    def update_data(self, i):
        self.text.remove()
        self.text2.remove()
        self.data = self.stream.read(self.CHUNK)
        self.data_int = struct.unpack(str(2 * self.CHUNK) + 'B', self.data)
        self.data_np = np.array(self.data_int, dtype='b')[::2] + 128
        self.yf = fft(self.data_int)
        self.X_mag = np.abs(self.yf[0:self.CHUNK])
        self.X_mag_plot = (2*self.X_mag[:int(len(self.X_mag)/2)])/(256*self.CHUNK)
        self.xpeak = np.argmax(self.X_mag_plot[4:])
        self.ypeak = np.max(self.X_mag_plot[4:])
        self.Ymark = self.X_mag_plot[self.index]
        self.text = self.ax2.text(4000, 0.9, f"Peak: {round(self.FREQ[self.xpeak+4],2)}Hz, {round(self.ypeak,4)}V")
        self.text2 = self.ax2.text(8000, 0.9, f"Marker: {round(self.Xmark,2)}Hz, {round(self.Ymark,2)}V")
        self.Oscope.set_data(self.xdata, self.data_np)
        self.SpecAn.set_data(self.FREQ, self.X_mag_plot)
        self.Peak.set_data(self.FREQ[self.xpeak+4], self.ypeak)
        self.Marker.set_data(self.Xmark, self.Ymark)

    def onPress(self,event):
        if self.ani_run == True:
            self.single_shot()
            self.ani.event_source.stop()
            self.ani_run = False

        else:
            self.ani.event_source.start()
            self.ani_run = True

    def mouse_move(self, event):
        try:
            if event.xdata is not None:
                self.Xmark = event.xdata
                y = np.flatnonzero(self.FREQ<=self.Xmark)
                self.index = len(y)-1
        
            if self.ani_run == False:
                self.single_shot()
        except:
            pass

    def single_shot(self):
        self.text.remove()
        self.text2.remove()
        self.Ymark = self.X_mag_plot[self.index]
        self.Marker.set_data(self.Xmark, self.Ymark)
        self.fig.canvas.draw_idle()
        self.text = self.ax2.text(self.FREQ[self.xpeak+4]+20, self.ypeak +0.1 , f"Peak: {round(self.FREQ[self.xpeak+4],2)}Hz, {round(self.ypeak,4)}V") 
        self.text2 = self.ax2.text(self.Xmark+20, self.Ymark +0.1, f"Marker: {round(self.Xmark, 2)}Hz, {round(self.Ymark, 2)}V")

    def animate(self):
        self.create_graph()
        self.fig.canvas.mpl_connect('key_press_event', self.onPress)
        self.fig.canvas.mpl_connect('motion_notify_event', self.mouse_move)
        self.ani = FuncAnimation(self.fig, self.update_data, interval=30, blit=False, repeat=True)
        self.ani_run = True
        plt.show()

audio = AudioStream()
audio.animate()




