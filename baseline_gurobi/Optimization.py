import math
from gurobipy import Model, GRB, quicksum
from gurobipy import nlfunc

class Optimization:

    def __init__(self,ins):

        self.ins = ins        
        self.model = None
        self.f = None
        self.q = None
        self.theta = None
        self.w = {i:1 for i in self.ins.A}

    def solveMINLP(self):
        
        self.model = Model("ACRP_MINLP")
        self.model.Params.NonConvex = 2  # Enable nonconvex MINLP
        self.model.setParam('OutputFlag', 0)
        
        # Variables
        f = self.model.addVars(self.ins.A, vtype=GRB.BINARY, name="f")
        q = self.model.addVars(self.ins.A, lb=self.ins.qmin, ub=self.ins.qmax, name="q")
        theta = self.model.addVars(self.ins.A, lb=self.ins.hmin, ub=self.ins.hmax, name="theta")        
        vrx = self.model.addVars(self.ins.P, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="vrx")
        vry = self.model.addVars(self.ins.P, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="vry")
        z = self.model.addVars(self.ins.P, vtype=GRB.BINARY, name="z")        
        
        tt = self.model.addVars(self.ins.A, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="tt")
        costheta = self.model.addVars(self.ins.A, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="costheta")
        sintheta = self.model.addVars(self.ins.A, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="sintheta")
        
        # Objective
        self.model.setObjective(quicksum(self.w[i]*f[i] for i in self.ins.A),GRB.MINIMIZE)
        #self.model.setObjective(quicksum(f[i] + q[i] + theta[i] for i in self.ins.A),GRB.MINIMIZE)

        # Constraints
        for i in self.ins.A:
            self.model.addConstr(f[i]*self.ins.qmin + (1 - f[i]) <= q[i])
            self.model.addConstr(f[i]*self.ins.qmax + (1 - f[i]) >= q[i])
            self.model.addConstr(f[i]*self.ins.hmin <= theta[i])
            self.model.addConstr(f[i]*self.ins.hmax >= theta[i])
            
            self.model.addConstr(tt[i] == self.ins.theta0[i] + theta[i])            
            self.model.addConstr(costheta[i] == nlfunc.cos(tt[i]))
            self.model.addConstr(sintheta[i] == nlfunc.sin(tt[i]))    
        
        for (i, j) in self.ins.P:
            # Velocity difference constraints
            self.model.addConstr(vrx[(i, j)] == q[i]*self.ins.v0[i]*costheta[i] - q[j]*self.ins.v0[j]*costheta[j],name=f"cvrx_{i}_{j}")
            self.model.addConstr(vry[(i, j)] == q[i]*self.ins.v0[i]*sintheta[i] - q[j]*self.ins.v0[j]*sintheta[j],name=f"cvry_{i}_{j}")

            # bin11 and bin12
            self.model.addConstr(self.ins.xr0[(i, j)]*vry[(i, j)] - self.ins.yr0[(i, j)]*vrx[(i, j)] <= (1 - z[(i, j)])*self.ins.Mbin11[(i, j)],name=f"bin11_{i}_{j}")
            self.model.addConstr(self.ins.xr0[(i, j)]*vry[(i, j)] - self.ins.yr0[(i, j)]*vrx[(i, j)] >= -z[(i, j)]*self.ins.Mbin12[(i, j)],name=f"bin12_{i}_{j}")
            
            # bin311 and bin312 (xr0>=0, yr0<0)
            if self.ins.xr0[(i, j)] >= 0 and self.ins.yr0[(i, j)] < 0:
                self.model.addConstr(self.ins.gammal[(i, j)]*vrx[(i, j)] - self.ins.phil[(i, j)]*vry[(i, j)] <= (1 - z[(i, j)])*self.ins.Mbin311[(i, j)],name=f"bin311_{i}_{j}")
                self.model.addConstr(self.ins.gammau[(i, j)]*vry[(i, j)] + self.ins.phiu[(i, j)]*vrx[(i, j)] <= z[(i, j)]*self.ins.Mbin312[(i, j)],name=f"bin312_{i}_{j}")

            # bin321 and bin322 (xr0<0, yr0>=0)
            if self.ins.xr0[(i, j)] < 0 and self.ins.yr0[(i, j)] >= 0:
                self.model.addConstr(self.ins.gammal[(i, j)]*vrx[(i, j)] + self.ins.phil[(i, j)]*vry[(i, j)] <= (1 - z[(i, j)])*self.ins.Mbin321[(i, j)],name=f"bin321_{i}_{j}")
                self.model.addConstr(self.ins.gammau[(i, j)]*vry[(i, j)] - self.ins.phiu[(i, j)]*vrx[(i, j)] <= z[(i, j)]*self.ins.Mbin322[(i, j)],name=f"bin322_{i}_{j}")

            # bin331 and bin332 (xr0>=0, yr0>=0)
            if self.ins.xr0[(i, j)] >= 0 and self.ins.yr0[(i, j)] >= 0:
                self.model.addConstr(self.ins.gammal[(i, j)]*vry[(i, j)] - self.ins.phil[(i, j)]*vrx[(i, j)] <= (1 - z[(i, j)])*self.ins.Mbin331[(i, j)],name=f"bin331_{i}_{j}")
                self.model.addConstr(self.ins.gammau[(i, j)]*vrx[(i, j)] - self.ins.phiu[(i, j)]*vry[(i, j)] <= z[(i, j)]*self.ins.Mbin332[(i, j)],name=f"bin332_{i}_{j}")

            # bin341 and bin342 (xr0<0, yr0<0)
            if self.ins.xr0[(i, j)] < 0 and self.ins.yr0[(i, j)] < 0:
                self.model.addConstr(self.ins.gammal[(i, j)]*vry[(i, j)] + self.ins.phil[(i, j)]*vrx[(i, j)] <= (1 - z[(i, j)])*self.ins.Mbin341[(i, j)],name=f"bin341_{i}_{j}")
                self.model.addConstr(self.ins.gammau[(i, j)]*vrx[(i, j)] + self.ins.phiu[(i, j)]*vry[(i, j)] <= z[(i, j)]*self.ins.Mbin342[(i, j)],name=f"bin342_{i}_{j}")
            
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL or self.model.status == GRB.SUBOPTIMAL:
            print("OFV: %d" % self.model.ObjVal)
            
            self.f = {i:f[i].X for i in self.ins.A}
            self.q = {i:q[i].X for i in self.ins.A}
            self.theta = {i:theta[i].X for i in self.ins.A}            
            
            for i in self.ins.A:
                print("%d\t%d\t%.3f\t%.3f" % (i,self.f[i],self.q[i],self.theta[i]))
        else:
            print("No feasible solution found.")        
            
    def oracle(self,Fmin):
        
        T = range(1,self.ins.T+1)
        
        oracle = Model("ACRP_ORACLE")
        oracle.Params.NonConvex = 2  # Enable nonconvex MINLP
        oracle.setParam('OutputFlag', 0)
        
        # Variables
        f = oracle.addVars(T,self.ins.A, vtype=GRB.BINARY, name="f")
        q = oracle.addVars(T,self.ins.A, lb=self.ins.qmin, ub=self.ins.qmax, name="q")
        theta = oracle.addVars(T,self.ins.A, lb=self.ins.hmin, ub=self.ins.hmax, name="theta")        
        vrx = oracle.addVars(T,self.ins.P, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="vrx")
        vry = oracle.addVars(T,self.ins.P, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="vry")
        z = oracle.addVars(T,self.ins.P, vtype=GRB.BINARY, name="z")        
        
        tt = oracle.addVars(T,self.ins.A, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="tt")
        costheta = oracle.addVars(T,self.ins.A, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="costheta")
        sintheta = oracle.addVars(T,self.ins.A, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="sintheta")
        
        dev = oracle.addVars(self.ins.A, lb=0.0, ub=GRB.INFINITY, name="dev")
        
        # Objective
        oracle.setObjective(quicksum(dev[i] for i in self.ins.A),GRB.MINIMIZE)
        
        # Constraints
        for i in self.ins.A:
            oracle.addConstr(dev[i] == nlfunc.abs(quicksum(f[t,i] for t in T) - Fmin/self.ins.n))
        
        for t in T:
                
            for i in self.ins.A:
                self.ins.v0[i] = self.ins.v0rand[t][i]
            self.ins.preprocessing()
            
            # Constraints
            for i in self.ins.A:
                oracle.addConstr(f[t,i]*self.ins.qmin + (1 - f[t,i]) <= q[t,i])
                oracle.addConstr(f[t,i]*self.ins.qmax + (1 - f[t,i]) >= q[t,i])
                oracle.addConstr(f[t,i]*self.ins.hmin <= theta[t,i])
                oracle.addConstr(f[t,i]*self.ins.hmax >= theta[t,i])
                
                oracle.addConstr(tt[t,i] == self.ins.theta0[i] + theta[t,i])            
                oracle.addConstr(costheta[t,i] == nlfunc.cos(tt[t,i]))
                oracle.addConstr(sintheta[t,i] == nlfunc.sin(tt[t,i]))    
            
            for (i, j) in self.ins.P:
                # Velocity difference constraints
                oracle.addConstr(vrx[t,(i, j)] == q[t,i]*self.ins.v0[i]*costheta[t,i] - q[t,j]*self.ins.v0[j]*costheta[t,j])
                oracle.addConstr(vry[t,(i, j)] == q[t,i]*self.ins.v0[i]*sintheta[t,i] - q[t,j]*self.ins.v0[j]*sintheta[t,j])

                # bin11 and bin12
                oracle.addConstr(self.ins.xr0[(i, j)]*vry[t,(i, j)] - self.ins.yr0[(i, j)]*vrx[t,(i, j)] <= (1 - z[t,(i, j)])*self.ins.Mbin11[(i, j)])
                oracle.addConstr(self.ins.xr0[(i, j)]*vry[t,(i, j)] - self.ins.yr0[(i, j)]*vrx[t,(i, j)] >= -z[t,(i, j)]*self.ins.Mbin12[(i, j)])
                
                # bin311 and bin312 (xr0>=0, yr0<0)
                if self.ins.xr0[(i, j)] >= 0 and self.ins.yr0[(i, j)] < 0:
                    oracle.addConstr(self.ins.gammal[(i, j)]*vrx[t,(i, j)] - self.ins.phil[(i, j)]*vry[t,(i, j)] <= (1 - z[t,(i, j)])*self.ins.Mbin311[(i, j)])
                    oracle.addConstr(self.ins.gammau[(i, j)]*vry[t,(i, j)] + self.ins.phiu[(i, j)]*vrx[t,(i, j)] <= z[t,(i, j)]*self.ins.Mbin312[(i, j)])

                # bin321 and bin322 (xr0<0, yr0>=0)
                if self.ins.xr0[(i, j)] < 0 and self.ins.yr0[(i, j)] >= 0:
                    oracle.addConstr(self.ins.gammal[(i, j)]*vrx[t,(i, j)] + self.ins.phil[(i, j)]*vry[t,(i, j)] <= (1 - z[t,(i, j)])*self.ins.Mbin321[(i, j)])
                    oracle.addConstr(self.ins.gammau[(i, j)]*vry[t,(i, j)] - self.ins.phiu[(i, j)]*vrx[t,(i, j)] <= z[t,(i, j)]*self.ins.Mbin322[(i, j)])

                # bin331 and bin332 (xr0>=0, yr0>=0)
                if self.ins.xr0[(i, j)] >= 0 and self.ins.yr0[(i, j)] >= 0:
                    oracle.addConstr(self.ins.gammal[(i, j)]*vry[t,(i, j)] - self.ins.phil[(i, j)]*vrx[t,(i, j)] <= (1 - z[t,(i, j)])*self.ins.Mbin331[(i, j)])
                    oracle.addConstr(self.ins.gammau[(i, j)]*vrx[t,(i, j)] - self.ins.phiu[(i, j)]*vry[t,(i, j)] <= z[t,(i, j)]*self.ins.Mbin332[(i, j)])

                # bin341 and bin342 (xr0<0, yr0<0)
                if self.ins.xr0[(i, j)] < 0 and self.ins.yr0[(i, j)] < 0:
                    oracle.addConstr(self.ins.gammal[(i, j)]*vry[t,(i, j)] + self.ins.phil[(i, j)]*vrx[t,(i, j)] <= (1 - z[t,(i, j)])*self.ins.Mbin341[(i, j)])
                    oracle.addConstr(self.ins.gammau[(i, j)]*vrx[t,(i, j)] + self.ins.phiu[(i, j)]*vry[t,(i, j)] <= z[t,(i, j)]*self.ins.Mbin342[(i, j)])
                    
        oracle.optimize()

        if oracle.status == GRB.OPTIMAL or oracle.status == GRB.SUBOPTIMAL:
            print("OFV: %d" % oracle.ObjVal)
            
            f = {(t,i):f[t,i].X for t in T for i in self.ins.A}
                        
            for i in self.ins.A:
                print("%d\t%d\t%d" % (t,i,f[t,i]))
        else:
            print("No feasible solution found.")                           