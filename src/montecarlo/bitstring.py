import numpy as np
import math

class BitString:
    def __init__(self, N):
        self.N = N
        self.config = np.zeros(N, dtype=int) 

    def __repr__(self):
        out = ""
        for i in self.config:
            out += str(i)
        return out

    def __eq__(self, other):        
        return all(self.config == other.config)
    
    def __len__(self):
        return len(self.config)

    def on(self):
        return self.config.count(1)

    def off(self):
        return self.config.count(0)

    def flip_site(self,i):
        if self.config[i] == 0:
            self.config[i] = 1
        else:
            self.config[i] = 0

    
    def integer(self):
        place = len(self.config) - 1
        number = 0
        for i in self.config:
            if i == 1:
                number = math.pow(2, place) + number
            place-=1
        return number 
 

    def set_config(self, s:list[int]):
        self.config = s

    def set_integer_config(self, dec:int):
        self.config = np.zeros(self.N, dtype=int)
        num = dec
        place = self.N - 1
        while num != 0:
            self.config[place] = num % 2
            place-=1
            num//= 2
        return self.config