import time

class PID:
    def __init__(self, kP=1, kI=0, kD=0):
        self.kP = kP
        self.kI = kI
        self.kD = kD
        
    def initialize(self):
        self.currTime = time.time()
        self.prevTime = self.currTime
        
        self.prevError = 0
        
        self.cP = 0
        self.cI = 0
        self.cD = 0
        
    
    def update(self, error, sleep=0.2):
        time.sleep(sleep)
        
        self.currTime = time.time()
        deltaTime = self.currTime - self.prevTime
        
        deltaError = error - self.prevError
        
        self.cP = error
        self.cI += error * deltaTime
        self.cD = (deltaError / deltaTime) if deltaTime > 0 else 0
        
        self.prevTime = self.currTime
        self.prevError = error
        
        
        return sum([
            self.kP * self.cP,
            self.kI * self.cI,
            self.kD * self.cD])
        