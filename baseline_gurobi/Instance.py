import math

class Instance:

    def __init__(self,filename,type):

        self.n = 0
        self.d = 0
        self.radius = 0
        self.v0base = {}   
        self.v0 = {}             
        self.cap = {}
        self.A = []
        self.P = []
        self.theta0 = {}
        self.x0 = {}
        self.y0 = {}
        self.xr0 = {}
        self.yr0 = {}        
        self.qmin = 0.94
        self.qmax = 1.03
        self.hmin = -math.pi/6
        self.hmax = math.pi/6
        self.ucc = 1
        self.lcc = min(math.cos(self.hmin), math.cos(self.hmax))
        self.uss = math.sin(self.hmax)
        self.lss = math.sin(self.hmin)        

        self.gammal = {}
        self.gammau = {}
        self.phil = {}
        self.phiu = {}
        self.Mbin11 = {}
        self.Mbin12 = {}
        self.Mbin311 = {}
        self.Mbin312 = {}
        self.Mbin321 = {}
        self.Mbin322 = {}
        self.Mbin331 = {} 
        self.Mbin332 = {}
        self.Mbin341 = {}
        self.Mbin342 = {}
        self.uvrx = {}
        self.uvry = {}
        self.lvrx = {}
        self.lvry = {}
        
        self.eps = None
        self.T = None
        self.v0rand = {}        
        self.Fmin = {}        
        
        if type == "static":
            self.read_static_instance(filename)
        
        if type == "dynamic":
            self.read_dynamic_instance(filename)
                    
    def read_static_instance(self,filename):

        xy_data = False
        with open(filename, 'r') as f:
            for line in f:
                if 'param d' in line:
                    self.d = float(line.split(':=')[1].strip('; \n'))
                elif 'param n' in line:
                    self.n = int(line.split(':=')[1].strip('; \n'))
                elif 'param radius' in line:
                    self.radius = float(line.split(':=')[1].strip('; \n'))
                elif 'param v0' in line:
                    for _ in range(self.n):
                        idx, val = f.readline().split()
                        self.v0[int(idx)] = float(val)
                        self.v0base[int(idx)] = float(val)
                elif 'param cap' in line:
                    for _ in range(self.n):
                        idx, val = f.readline().split()
                        self.cap[int(idx)] = float(val)  
                elif 'param x0' in line:
                    xy_data = True
                    for _ in range(self.n):
                        idx, val = f.readline().split()
                        self.x0[int(idx)] = float(val)
                elif 'param y0' in line:
                    for _ in range(self.n):
                        idx, val = f.readline().split()
                        self.y0[int(idx)] = float(val)                        
        
        self.A = [i for i in range(1,self.n+1)]
        self.P = [(i, j) for i in self.A for j in self.A if i < j]
        
        if xy_data == False:
            #---code specific to CP instances
            for i in self.A:
                self.x0[i] = -self.radius * math.cos((i-1)*2*math.pi/self.n + math.pi)
                self.y0[i] = -self.radius * math.sin((i-1)*2*math.pi/self.n + math.pi)                        
                
        for i in self.A:
            self.theta0[i] = self.cap[i] - 2*math.pi if self.cap[i] >= math.pi else self.cap[i]

        for i, j in self.P:
            self.xr0[(i, j)] = self.x0[i] - self.x0[j]
            self.yr0[(i, j)] = self.y0[i] - self.y0[j]  
        
    def read_dynamic_instance(self,filename):

        with open(filename, 'r') as f:
            for line in f:
                if 'param d' in line:
                    self.d = float(line.split(':=')[1].strip('; \n'))
                elif 'param n' in line:
                    self.n = int(line.split(':=')[1].strip('; \n'))
                elif 'param radius' in line:
                    self.radius = float(line.split(':=')[1].strip('; \n'))
                elif 'param T' in line:
                    self.T = int(line.split(':=')[1].strip('; \n'))
                    self.v0rand = {t:{} for t in range(1,self.T+1)}
                    self.P0rand = {t:[] for t in range(1,self.T+1)}
                elif 'param eps' in line:
                    self.eps = float(line.split(':=')[1].strip('; \n'))                    
                elif 'param v0' in line:
                    if 'param v0rand' in line:
                        for _ in range(self.T):
                            for _ in range(self.n):
                                t, i, val = f.readline().split()
                                self.v0rand[int(t)][int(i)] = float(val)                        
                    else:
                        for _ in range(self.n):
                            idx, val = f.readline().split()
                            self.v0base[int(idx)] = float(val)
                elif 'param cap' in line:
                    for _ in range(self.n):
                        idx, val = f.readline().split()
                        self.cap[int(idx)] = float(val)  
                elif 'param x0' in line:
                    for _ in range(self.n):
                        idx, val = f.readline().split()
                        self.x0[int(idx)] = float(val)
                elif 'param y0' in line:
                    for _ in range(self.n):
                        idx, val = f.readline().split()
                        self.y0[int(idx)] = float(val)
        
        self.A = [i for i in range(1,self.n+1)]
        self.P = [(i, j) for i in self.A for j in self.A if i < j]
        
        for i in self.A:
            self.theta0[i] = self.cap[i] - 2*math.pi if self.cap[i] >= math.pi else self.cap[i]

        for i, j in self.P:
            self.xr0[(i, j)] = self.x0[i] - self.x0[j]
            self.yr0[(i, j)] = self.y0[i] - self.y0[j]  
                
    def conflict_detection(self, t):
        
        if t == 0:
            for i in self.A:
                self.v0[i] = self.v0base[i]
        else:
            for i in self.A:
                self.v0[i] = self.v0rand[t][i]
        
        conflicts = []
        for (i,j) in self.P:
            vrx0 = self.v0[i]*math.cos(self.theta0[i]) - self.v0[j]*math.cos(self.theta0[j])
            vry0 = self.v0[i]*math.sin(self.theta0[i]) - self.v0[j]*math.sin(self.theta0[j])
            sep0 = (self.yr0[i,j]**2 - self.d**2)*vrx0**2 + (self.xr0[i,j]**2 - self.d**2)*vry0**2 - 2*self.xr0[i,j]*self.yr0[i,j]*vrx0*vry0
            tm0 = -(self.xr0[i,j]*vrx0 + self.yr0[i,j]*vry0)/(vrx0**2 + vry0**2)
            if sep0 < 0 and tm0 > 0:
                conflicts.append((i,j))                  
        
        return conflicts
    
    def preprocessing(self):
        
        #---uses v0---#

        a, b, c = {}, {}, {}

        for (i, j) in self.P:
            ci, cj = math.cos(self.theta0[i]), math.cos(self.theta0[j])
            si, sj = math.sin(self.theta0[i]), math.sin(self.theta0[j])

            # Bounds for vrx and vry 
            if ci >= 0 and cj >= 0 and si >= 0 and sj >= 0:
                self.uvrx[i, j] = self.qmax*self.ucc*self.v0[i]*ci - self.qmin*self.lss*self.v0[i]*si - self.qmin*self.lcc*self.v0[j]*cj + self.qmax*self.uss*self.v0[j]*sj
                self.lvrx[i, j] = self.qmin*self.lcc*self.v0[i]*ci - self.qmax*self.uss*self.v0[i]*si - self.qmax*self.ucc*self.v0[j]*cj + self.qmin*self.lss*self.v0[j]*sj
                self.uvry[i, j] = self.qmax*self.uss*self.v0[i]*ci + self.qmax*self.ucc*self.v0[i]*si - self.qmin*self.lss*self.v0[j]*cj - self.qmin*self.lcc*self.v0[j]*sj
                self.lvry[i, j] = self.qmin*self.lss*self.v0[i]*ci + self.qmin*self.lcc*self.v0[i]*si - self.qmax*self.uss*self.v0[j]*cj - self.qmax*self.ucc*self.v0[j]*sj
            if ci >= 0 and cj >= 0 and si >= 0 and sj < 0:
                self.uvrx[i, j] = self.qmax*self.ucc*self.v0[i]*ci - self.qmin*self.lss*self.v0[i]*si - self.qmin*self.lcc*self.v0[j]*cj + self.qmin*self.lss*self.v0[j]*sj
                self.lvrx[i, j] = self.qmin*self.lcc*self.v0[i]*ci - self.qmax*self.uss*self.v0[i]*si - self.qmax*self.ucc*self.v0[j]*cj + self.qmax*self.uss*self.v0[j]*sj
                self.uvry[i, j] = self.qmax*self.uss*self.v0[i]*ci + self.qmax*self.ucc*self.v0[i]*si - self.qmin*self.lss*self.v0[j]*cj - self.qmax*self.ucc*self.v0[j]*sj
                self.lvry[i, j] = self.qmin*self.lss*self.v0[i]*ci + self.qmin*self.lcc*self.v0[i]*si - self.qmax*self.uss*self.v0[j]*cj - self.qmin*self.lcc*self.v0[j]*sj
            if ci >= 0 and cj >= 0 and si < 0 and sj >= 0:
                self.uvrx[i, j] = self.qmax*self.ucc*self.v0[i]*ci - self.qmax*self.uss*self.v0[i]*si - self.qmin*self.lcc*self.v0[j]*cj + self.qmax*self.uss*self.v0[j]*sj
                self.lvrx[i, j] = self.qmin*self.lcc*self.v0[i]*ci - self.qmin*self.lss*self.v0[i]*si - self.qmax*self.ucc*self.v0[j]*cj + self.qmin*self.lss*self.v0[j]*sj
                self.uvry[i, j] = self.qmax*self.uss*self.v0[i]*ci + self.qmin*self.lcc*self.v0[i]*si - self.qmin*self.lss*self.v0[j]*cj - self.qmin*self.lcc*self.v0[j]*sj
                self.lvry[i, j] = self.qmin*self.lss*self.v0[i]*ci + self.qmax*self.ucc*self.v0[i]*si - self.qmax*self.uss*self.v0[j]*cj - self.qmax*self.ucc*self.v0[j]*sj
            if ci >= 0 and cj >= 0 and si < 0 and sj < 0:
                self.uvrx[i, j] = self.qmax*self.ucc*self.v0[i]*ci - self.qmax*self.uss*self.v0[i]*si - self.qmin*self.lcc*self.v0[j]*cj + self.qmin*self.lss*self.v0[j]*sj
                self.lvrx[i, j] = self.qmin*self.lcc*self.v0[i]*ci - self.qmin*self.lss*self.v0[i]*si - self.qmax*self.ucc*self.v0[j]*cj + self.qmax*self.uss*self.v0[j]*sj
                self.uvry[i, j] = self.qmax*self.uss*self.v0[i]*ci + self.qmin*self.lcc*self.v0[i]*si - self.qmin*self.lss*self.v0[j]*cj - self.qmax*self.ucc*self.v0[j]*sj
                self.lvry[i, j] = self.qmin*self.lss*self.v0[i]*ci + self.qmax*self.ucc*self.v0[i]*si - self.qmax*self.uss*self.v0[j]*cj - self.qmin*self.lcc*self.v0[j]*sj
            if ci >= 0 and cj < 0 and si >= 0 and sj >= 0:
                self.uvrx[i, j] = self.qmax*self.ucc*self.v0[i]*ci - self.qmin*self.lss*self.v0[i]*si - self.qmax*self.ucc*self.v0[j]*cj + self.qmax*self.uss*self.v0[j]*sj
                self.lvrx[i, j] = self.qmin*self.lcc*self.v0[i]*ci - self.qmax*self.uss*self.v0[i]*si - self.qmin*self.lcc*self.v0[j]*cj + self.qmin*self.lss*self.v0[j]*sj
                self.uvry[i, j] = self.qmax*self.uss*self.v0[i]*ci + self.qmax*self.ucc*self.v0[i]*si - self.qmax*self.uss*self.v0[j]*cj - self.qmin*self.lcc*self.v0[j]*sj
                self.lvry[i, j] = self.qmin*self.lss*self.v0[i]*ci + self.qmin*self.lcc*self.v0[i]*si - self.qmin*self.lss*self.v0[j]*cj - self.qmax*self.ucc*self.v0[j]*sj
            if ci >= 0 and cj < 0 and si >= 0 and sj < 0:
                self.uvrx[i, j] = self.qmax*self.ucc*self.v0[i]*ci - self.qmin*self.lss*self.v0[i]*si - self.qmax*self.ucc*self.v0[j]*cj + self.qmin*self.lss*self.v0[j]*sj
                self.lvrx[i, j] = self.qmin*self.lcc*self.v0[i]*ci - self.qmax*self.uss*self.v0[i]*si - self.qmin*self.lcc*self.v0[j]*cj + self.qmax*self.uss*self.v0[j]*sj
                self.uvry[i, j] = self.qmax*self.uss*self.v0[i]*ci + self.qmax*self.ucc*self.v0[i]*si - self.qmax*self.uss*self.v0[j]*cj - self.qmax*self.ucc*self.v0[j]*sj
                self.lvry[i, j] = self.qmin*self.lss*self.v0[i]*ci + self.qmin*self.lcc*self.v0[i]*si - self.qmin*self.lss*self.v0[j]*cj - self.qmin*self.lcc*self.v0[j]*sj
            if ci >= 0 and cj < 0 and si < 0 and sj >= 0:
                self.uvrx[i, j] = self.qmax*self.ucc*self.v0[i]*ci - self.qmax*self.uss*self.v0[i]*si - self.qmax*self.ucc*self.v0[j]*cj + self.qmax*self.uss*self.v0[j]*sj
                self.lvrx[i, j] = self.qmin*self.lcc*self.v0[i]*ci - self.qmin*self.lss*self.v0[i]*si - self.qmin*self.lcc*self.v0[j]*cj + self.qmin*self.lss*self.v0[j]*sj
                self.uvry[i, j] = self.qmax*self.uss*self.v0[i]*ci + self.qmin*self.lcc*self.v0[i]*si - self.qmax*self.uss*self.v0[j]*cj - self.qmin*self.lcc*self.v0[j]*sj
                self.lvry[i, j] = self.qmin*self.lss*self.v0[i]*ci + self.qmax*self.ucc*self.v0[i]*si - self.qmin*self.lss*self.v0[j]*cj - self.qmax*self.ucc*self.v0[j]*sj
            if ci >= 0 and cj < 0 and si < 0 and sj < 0:
                self.uvrx[i, j] = self.qmax*self.ucc*self.v0[i]*ci - self.qmax*self.uss*self.v0[i]*si - self.qmax*self.ucc*self.v0[j]*cj + self.qmin*self.lss*self.v0[j]*sj
                self.lvrx[i, j] = self.qmin*self.lcc*self.v0[i]*ci - self.qmin*self.lss*self.v0[i]*si - self.qmin*self.lcc*self.v0[j]*cj + self.qmax*self.uss*self.v0[j]*sj
                self.uvry[i, j] = self.qmax*self.uss*self.v0[i]*ci + self.qmin*self.lcc*self.v0[i]*si - self.qmax*self.uss*self.v0[j]*cj - self.qmax*self.ucc*self.v0[j]*sj
                self.lvry[i, j] = self.qmin*self.lss*self.v0[i]*ci + self.qmax*self.ucc*self.v0[i]*si - self.qmin*self.lss*self.v0[j]*cj - self.qmin*self.lcc*self.v0[j]*sj
            if ci < 0 and cj >= 0 and si >= 0 and sj >= 0:
                self.uvrx[i, j] = self.qmin*self.lcc*self.v0[i]*ci - self.qmin*self.lss*self.v0[i]*si - self.qmin*self.lcc*self.v0[j]*cj + self.qmax*self.uss*self.v0[j]*sj
                self.lvrx[i, j] = self.qmax*self.ucc*self.v0[i]*ci - self.qmax*self.uss*self.v0[i]*si - self.qmax*self.ucc*self.v0[j]*cj + self.qmin*self.lss*self.v0[j]*sj
                self.uvry[i, j] = self.qmin*self.lss*self.v0[i]*ci + self.qmax*self.ucc*self.v0[i]*si - self.qmin*self.lss*self.v0[j]*cj - self.qmin*self.lcc*self.v0[j]*sj
                self.lvry[i, j] = self.qmax*self.uss*self.v0[i]*ci + self.qmin*self.lcc*self.v0[i]*si - self.qmax*self.uss*self.v0[j]*cj - self.qmax*self.ucc*self.v0[j]*sj
            if ci < 0 and cj >= 0 and si >= 0 and sj < 0:
                self.uvrx[i, j] = self.qmin*self.lcc*self.v0[i]*ci - self.qmin*self.lss*self.v0[i]*si - self.qmin*self.lcc*self.v0[j]*cj + self.qmin*self.lss*self.v0[j]*sj
                self.lvrx[i, j] = self.qmax*self.ucc*self.v0[i]*ci - self.qmax*self.uss*self.v0[i]*si - self.qmax*self.ucc*self.v0[j]*cj + self.qmax*self.uss*self.v0[j]*sj
                self.uvry[i, j] = self.qmin*self.lss*self.v0[i]*ci + self.qmax*self.ucc*self.v0[i]*si - self.qmin*self.lss*self.v0[j]*cj - self.qmax*self.ucc*self.v0[j]*sj
                self.lvry[i, j] = self.qmax*self.uss*self.v0[i]*ci + self.qmin*self.lcc*self.v0[i]*si - self.qmax*self.uss*self.v0[j]*cj - self.qmin*self.lcc*self.v0[j]*sj
            if ci < 0 and cj >= 0 and si < 0 and sj >= 0:
                self.uvrx[i, j] = self.qmin*self.lcc*self.v0[i]*ci - self.qmax*self.uss*self.v0[i]*si - self.qmin*self.lcc*self.v0[j]*cj + self.qmax*self.uss*self.v0[j]*sj
                self.lvrx[i, j] = self.qmax*self.ucc*self.v0[i]*ci - self.qmin*self.lss*self.v0[i]*si - self.qmax*self.ucc*self.v0[j]*cj + self.qmin*self.lss*self.v0[j]*sj
                self.uvry[i, j] = self.qmin*self.lss*self.v0[i]*ci + self.qmin*self.lcc*self.v0[i]*si - self.qmin*self.lss*self.v0[j]*cj - self.qmin*self.lcc*self.v0[j]*sj
                self.lvry[i, j] = self.qmax*self.uss*self.v0[i]*ci + self.qmax*self.ucc*self.v0[i]*si - self.qmax*self.uss*self.v0[j]*cj - self.qmax*self.ucc*self.v0[j]*sj
            if ci < 0 and cj >= 0 and si < 0 and sj < 0:
                self.uvrx[i, j] = self.qmin*self.lcc*self.v0[i]*ci - self.qmax*self.uss*self.v0[i]*si - self.qmin*self.lcc*self.v0[j]*cj + self.qmin*self.lss*self.v0[j]*sj
                self.lvrx[i, j] = self.qmax*self.ucc*self.v0[i]*ci - self.qmin*self.lss*self.v0[i]*si - self.qmax*self.ucc*self.v0[j]*cj + self.qmax*self.uss*self.v0[j]*sj
                self.uvry[i, j] = self.qmin*self.lss*self.v0[i]*ci + self.qmin*self.lcc*self.v0[i]*si - self.qmin*self.lss*self.v0[j]*cj - self.qmax*self.ucc*self.v0[j]*sj
                self.lvry[i, j] = self.qmax*self.uss*self.v0[i]*ci + self.qmax*self.ucc*self.v0[i]*si - self.qmax*self.uss*self.v0[j]*cj - self.qmin*self.lcc*self.v0[j]*sj                
            if ci < 0 and cj < 0 and si >= 0 and sj >= 0:
                self.uvrx[i, j] = self.qmin*self.lcc*self.v0[i]*ci - self.qmin*self.lss*self.v0[i]*si - self.qmax*self.ucc*self.v0[j]*cj + self.qmax*self.uss*self.v0[j]*sj
                self.lvrx[i, j] = self.qmax*self.ucc*self.v0[i]*ci - self.qmax*self.uss*self.v0[i]*si - self.qmin*self.lcc*self.v0[j]*cj + self.qmin*self.lss*self.v0[j]*sj
                self.uvry[i, j] = self.qmin*self.lss*self.v0[i]*ci + self.qmax*self.ucc*self.v0[i]*si - self.qmax*self.uss*self.v0[j]*cj - self.qmin*self.lcc*self.v0[j]*sj
                self.lvry[i, j] = self.qmax*self.uss*self.v0[i]*ci + self.qmin*self.lcc*self.v0[i]*si - self.qmin*self.lss*self.v0[j]*cj - self.qmax*self.ucc*self.v0[j]*sj	
            if ci < 0 and cj < 0 and si >= 0 and sj < 0:
                self.uvrx[i, j] = self.qmin*self.lcc*self.v0[i]*ci - self.qmin*self.lss*self.v0[i]*si - self.qmax*self.ucc*self.v0[j]*cj + self.qmin*self.lss*self.v0[j]*sj
                self.lvrx[i, j] = self.qmax*self.ucc*self.v0[i]*ci - self.qmax*self.uss*self.v0[i]*si - self.qmin*self.lcc*self.v0[j]*cj + self.qmax*self.uss*self.v0[j]*sj
                self.uvry[i, j] = self.qmin*self.lss*self.v0[i]*ci + self.qmax*self.ucc*self.v0[i]*si - self.qmax*self.uss*self.v0[j]*cj - self.qmax*self.ucc*self.v0[j]*sj
                self.lvry[i, j] = self.qmax*self.uss*self.v0[i]*ci + self.qmin*self.lcc*self.v0[i]*si - self.qmin*self.lss*self.v0[j]*cj - self.qmin*self.lcc*self.v0[j]*sj	
            if ci < 0 and cj < 0 and si < 0 and sj >= 0:
                self.uvrx[i, j] = self.qmin*self.lcc*self.v0[i]*ci - self.qmax*self.uss*self.v0[i]*si - self.qmax*self.ucc*self.v0[j]*cj + self.qmax*self.uss*self.v0[j]*sj
                self.lvrx[i, j] = self.qmax*self.ucc*self.v0[i]*ci - self.qmin*self.lss*self.v0[i]*si - self.qmin*self.lcc*self.v0[j]*cj + self.qmin*self.lss*self.v0[j]*sj
                self.uvry[i, j] = self.qmin*self.lss*self.v0[i]*ci + self.qmin*self.lcc*self.v0[i]*si - self.qmax*self.uss*self.v0[j]*cj - self.qmin*self.lcc*self.v0[j]*sj
                self.lvry[i, j] = self.qmax*self.uss*self.v0[i]*ci + self.qmax*self.ucc*self.v0[i]*si - self.qmin*self.lss*self.v0[j]*cj - self.qmax*self.ucc*self.v0[j]*sj
            if ci < 0 and cj < 0 and si < 0 and sj < 0:
                self.uvrx[i, j] = self.qmin*self.lcc*self.v0[i]*ci - self.qmax*self.uss*self.v0[i]*si - self.qmax*self.ucc*self.v0[j]*cj + self.qmin*self.lss*self.v0[j]*sj
                self.lvrx[i, j] = self.qmax*self.ucc*self.v0[i]*ci - self.qmin*self.lss*self.v0[i]*si - self.qmin*self.lcc*self.v0[j]*cj + self.qmax*self.uss*self.v0[j]*sj
                self.uvry[i, j] = self.qmin*self.lss*self.v0[i]*ci + self.qmin*self.lcc*self.v0[i]*si - self.qmax*self.uss*self.v0[j]*cj - self.qmax*self.ucc*self.v0[j]*sj
                self.lvry[i, j] = self.qmax*self.uss*self.v0[i]*ci + self.qmax*self.ucc*self.v0[i]*si - self.qmin*self.lss*self.v0[j]*cj - self.qmin*self.lcc*self.v0[j]*sj		             

            # a, b, c
            a[i, j] = self.yr0[i, j]**2 - self.d**2
            b[i, j] = self.xr0[i, j]**2 - self.d**2
            c[i, j] = 2 * self.xr0[i, j] * self.yr0[i, j]

            # Discriminant for sqrt
            disc = c[i, j]**2 - 4*a[i, j]*b[i, j]
            sqrt_disc = math.sqrt(disc) if disc >= 0 else 0

            # Mbin11, Mbin12
            if self.xr0[i, j] >= 0 and self.yr0[i, j] >= 0:
                self.Mbin11[i, j] = self.xr0[i, j]*self.uvry[i, j] - self.yr0[i, j]*self.lvrx[i, j]
                self.Mbin12[i, j] = -self.xr0[i, j]*self.lvry[i, j] + self.yr0[i, j]*self.uvrx[i, j]
            elif self.xr0[i, j] >= 0 and self.yr0[i, j] < 0:
                self.Mbin11[i, j] = self.xr0[i, j]*self.uvry[i, j] - self.yr0[i, j]*self.uvrx[i, j]
                self.Mbin12[i, j] = -self.xr0[i, j]*self.lvry[i, j] + self.yr0[i, j]*self.lvrx[i, j]
            elif self.xr0[i, j] < 0 and self.yr0[i, j] >= 0:
                self.Mbin11[i, j] = self.xr0[i, j]*self.lvry[i, j] - self.yr0[i, j]*self.lvrx[i, j]
                self.Mbin12[i, j] = -self.xr0[i, j]*self.uvry[i, j] + self.yr0[i, j]*self.uvrx[i, j]
            elif self.xr0[i, j] < 0 and self.yr0[i, j] < 0:
                self.Mbin11[i, j] = self.xr0[i, j]*self.lvry[i, j] - self.yr0[i, j]*self.uvrx[i, j]
                self.Mbin12[i, j] = -self.xr0[i, j]*self.uvry[i, j] + self.yr0[i, j]*self.lvrx[i, j]

            # (xr0>=0, yr0<0)
            if self.xr0[i, j] >= 0 and self.yr0[i, j] < 0:
                # Mbin311
                if a[i, j] >= 0 and -(c[i, j] - sqrt_disc) >= 0:
                    self.Mbin311[i, j] = 2*a[i, j]*self.uvrx[i, j] - self.uvry[i, j]*(c[i, j] - sqrt_disc)
                elif a[i, j] >= 0 and -(c[i, j] - sqrt_disc) < 0:
                    self.Mbin311[i, j] = 2*a[i, j]*self.uvrx[i, j] - self.lvry[i, j]*(c[i, j] - sqrt_disc)
                elif a[i, j] < 0 and -(c[i, j] - sqrt_disc) >= 0:
                    self.Mbin311[i, j] = 2*a[i, j]*self.lvrx[i, j] - self.uvry[i, j]*(c[i, j] - sqrt_disc)
                elif a[i, j] < 0 and -(c[i, j] - sqrt_disc) < 0:
                    self.Mbin311[i, j] = 2*a[i, j]*self.lvrx[i, j] - self.lvry[i, j]*(c[i, j] - sqrt_disc)
                # Mbin312
                if -b[i, j] >= 0 and (c[i, j] - sqrt_disc) >= 0:
                    self.Mbin312[i, j] = -2*b[i, j]*self.uvry[i, j] + self.uvrx[i, j]*(c[i, j] - sqrt_disc)
                elif -b[i, j] >= 0 and (c[i, j] - sqrt_disc) < 0:
                    self.Mbin312[i, j] = -2*b[i, j]*self.uvry[i, j] + self.lvrx[i, j]*(c[i, j] - sqrt_disc)
                elif -b[i, j] < 0 and (c[i, j] - sqrt_disc) >= 0:
                    self.Mbin312[i, j] = -2*b[i, j]*self.lvry[i, j] + self.uvrx[i, j]*(c[i, j] - sqrt_disc)
                elif -b[i, j] < 0 and (c[i, j] - sqrt_disc) < 0:
                    self.Mbin312[i, j] = -2*b[i, j]*self.lvry[i, j] + self.lvrx[i, j]*(c[i, j] - sqrt_disc)
                    
                self.gammal[i, j] = 2*a[i, j]
                self.gammau[i, j] = -2*b[i, j]
                self.phil[i, j] = c[i, j] - sqrt_disc
                self.phiu[i, j] = c[i, j] - sqrt_disc

            # (xr0<0, yr0>=0)
            if self.xr0[i, j] < 0 and self.yr0[i, j] >= 0:
                # Mbin321
                if -a[i, j] >= 0 and (c[i, j] - sqrt_disc) >= 0:
                    self.Mbin321[i, j] = -2*a[i, j]*self.uvrx[i, j] + self.uvry[i, j]*(c[i, j] - sqrt_disc)
                elif -a[i, j] >= 0 and (c[i, j] - sqrt_disc) < 0:
                    self.Mbin321[i, j] = -2*a[i, j]*self.uvrx[i, j] + self.lvry[i, j]*(c[i, j] - sqrt_disc)
                elif -a[i, j] < 0 and (c[i, j] - sqrt_disc) >= 0:
                    self.Mbin321[i, j] = -2*a[i, j]*self.lvrx[i, j] + self.uvry[i, j]*(c[i, j] - sqrt_disc)
                elif -a[i, j] < 0 and (c[i, j] - sqrt_disc) < 0:
                    self.Mbin321[i, j] = -2*a[i, j]*self.lvrx[i, j] + self.lvry[i, j]*(c[i, j] - sqrt_disc)
                # Mbin322
                if b[i, j] >= 0 and -(c[i, j] - sqrt_disc) >= 0:
                    self.Mbin322[i, j] = 2*b[i, j]*self.uvry[i, j] - self.uvrx[i, j]*(c[i, j] - sqrt_disc)
                elif b[i, j] >= 0 and -(c[i, j] - sqrt_disc) < 0:
                    self.Mbin322[i, j] = 2*b[i, j]*self.uvry[i, j] - self.lvrx[i, j]*(c[i, j] - sqrt_disc)
                elif b[i, j] < 0 and -(c[i, j] - sqrt_disc) >= 0:
                    self.Mbin322[i, j] = 2*b[i, j]*self.lvry[i, j] - self.uvrx[i, j]*(c[i, j] - sqrt_disc)
                elif b[i, j] < 0 and -(c[i, j] - sqrt_disc) < 0:
                    self.Mbin322[i, j] = 2*b[i, j]*self.lvry[i, j] - self.lvrx[i, j]*(c[i, j] - sqrt_disc)
                    
                self.gammal[i, j] = -2*a[i, j]
                self.gammau[i, j] = 2*b[i, j]
                self.phil[i, j] = c[i, j] - sqrt_disc
                self.phiu[i, j] = c[i, j] - sqrt_disc

            # (xr0>=0, yr0>=0)
            if self.xr0[i, j] >= 0 and self.yr0[i, j] >= 0:
                # Mbin331
                if b[i, j] >= 0 and -(c[i, j] + sqrt_disc) >= 0:
                    self.Mbin331[i, j] = 2*b[i, j]*self.uvry[i, j] - self.uvrx[i, j]*(c[i, j] + sqrt_disc)
                elif b[i, j] >= 0 and -(c[i, j] + sqrt_disc) < 0:
                    self.Mbin331[i, j] = 2*b[i, j]*self.uvry[i, j] - self.lvrx[i, j]*(c[i, j] + sqrt_disc)
                elif b[i, j] < 0 and -(c[i, j] + sqrt_disc) >= 0:
                    self.Mbin331[i, j] = 2*b[i, j]*self.lvry[i, j] - self.uvrx[i, j]*(c[i, j] + sqrt_disc)
                elif b[i, j] < 0 and -(c[i, j] + sqrt_disc) < 0:
                    self.Mbin331[i, j] = 2*b[i, j]*self.lvry[i, j] - self.lvrx[i, j]*(c[i, j] + sqrt_disc)
                # Mbin332
                if a[i, j] >= 0 and -(c[i, j] + sqrt_disc) >= 0:
                    self.Mbin332[i, j] = 2*a[i, j]*self.uvrx[i, j] - self.uvry[i, j]*(c[i, j] + sqrt_disc)
                elif a[i, j] >= 0 and -(c[i, j] + sqrt_disc) < 0:
                    self.Mbin332[i, j] = 2*a[i, j]*self.uvrx[i, j] - self.lvry[i, j]*(c[i, j] + sqrt_disc)
                elif a[i, j] < 0 and -(c[i, j] + sqrt_disc) >= 0:
                    self.Mbin332[i, j] = 2*a[i, j]*self.lvrx[i, j] - self.uvry[i, j]*(c[i, j] + sqrt_disc)
                elif a[i, j] < 0 and -(c[i, j] + sqrt_disc) < 0:
                    self.Mbin332[i, j] = 2*a[i, j]*self.lvrx[i, j] - self.lvry[i, j]*(c[i, j] + sqrt_disc)
                
                self.gammal[i, j] = 2*b[i, j]
                self.gammau[i, j] = 2*a[i, j]
                self.phil[i, j] = c[i, j] + sqrt_disc
                self.phiu[i, j] = c[i, j] + sqrt_disc

            # (xr0<0, yr0<0)
            if self.xr0[i, j] < 0 and self.yr0[i, j] < 0:
                # Mbin341
                if -b[i, j] >= 0 and (c[i, j] + sqrt_disc) >= 0:
                    self.Mbin341[i, j] = -2*b[i, j]*self.uvry[i, j] + self.uvrx[i, j]*(c[i, j] + sqrt_disc)
                elif -b[i, j] >= 0 and (c[i, j] + sqrt_disc) < 0:
                    self.Mbin341[i, j] = -2*b[i, j]*self.uvry[i, j] + self.lvrx[i, j]*(c[i, j] + sqrt_disc)
                elif -b[i, j] < 0 and (c[i, j] + sqrt_disc) >= 0:
                    self.Mbin341[i, j] = -2*b[i, j]*self.lvry[i, j] + self.uvrx[i, j]*(c[i, j] + sqrt_disc)
                elif -b[i, j] < 0 and (c[i, j] + sqrt_disc) < 0:
                    self.Mbin341[i, j] = -2*b[i, j]*self.lvry[i, j] + self.lvrx[i, j]*(c[i, j] + sqrt_disc)
                # Mbin342
                if -a[i, j] >= 0 and (c[i, j] + sqrt_disc) >= 0:
                    self.Mbin342[i, j] = -2*a[i, j]*self.uvrx[i, j] + self.uvry[i, j]*(c[i, j] + sqrt_disc)
                elif -a[i, j] >= 0 and (c[i, j] + sqrt_disc) < 0:
                    self.Mbin342[i, j] = -2*a[i, j]*self.uvrx[i, j] + self.lvry[i, j]*(c[i, j] + sqrt_disc)
                elif -a[i, j] < 0 and (c[i, j] + sqrt_disc) >= 0:
                    self.Mbin342[i, j] = -2*a[i, j]*self.lvrx[i, j] + self.uvry[i, j]*(c[i, j] + sqrt_disc)
                elif -a[i, j] < 0 and (c[i, j] + sqrt_disc) < 0:
                    self.Mbin342[i, j] = -2*a[i, j]*self.lvrx[i, j] + self.lvry[i, j]*(c[i, j] + sqrt_disc)
                    
                self.gammal[i, j] = -2*b[i, j]
                self.gammau[i, j] = -2*a[i, j]
                self.phil[i, j] = c[i, j] + sqrt_disc
                self.phiu[i, j] = c[i, j] + sqrt_disc
