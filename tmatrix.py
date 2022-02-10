#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 21:43:46 2022

@author: stephane
"""
import numpy as np
import pandas as pd

hbar = 6.625E-34/(2*np.pi)
m0 = 9.109E-31
eps0 = 8.85E-14
q = 1.602E-19

# define layer stack class


class stack:
    # The init method
    def __init__(self, wl):
        self.nair = 1.0*np.ones(wl.shape)
        self.nsubs = 1.5*np.ones(wl.shape)
        self.index = self.nsubs
        self.numlayers = int(0)
        self.wl=wl

    def addlayer(self, index, thickness):
        if self.numlayers == 0:
            self.thick = thickness
            self.index = np.vstack((index,self.index))
        else:
            self.thick = np.append(self.thick,thickness)          
            self.index = np.insert(self.index,self.numlayers,index,axis=0)

        if index.shape != self.nair.shape:
            raise Exception("Sorry, index has the wrong dimensions")
        self.numlayers += 1
    
    def addDBR(self, index1, index2, centerwl, pairno, hilo):
        thick1=centerwl/(4*np.interp(centerwl,self.wl,index1))
        thick2=centerwl/(4*np.interp(centerwl,self.wl,index2))
        if thick2>thick1:
            high=index1
            low=index2
        else:
            high=index2
            low=index1
        if hilo == 'HL':
            for k in range(2*pairno):
                if k % 2 == 1:
                    self.addlayer(low, max(thick1,thick2))
                else:
                    self.addlayer(high, min(thick1,thick2))
        elif hilo == 'LH':
            for k in range(2*pairno):
                if k % 2 == 0:
                    self.addlayer(low, max(thick1,thick2))
                else:
                    self.addlayer(high, min(thick1,thick2))
        else: raise Exception("Please specify DBR pair order")
    
    def calc(self,angles):
        # start calculations
        
        R_TE=np.empty((angles.shape[0],self.wl.shape[0]),dtype=float)
        R_TM=np.empty((angles.shape[0],self.wl.shape[0]),dtype=float)
        T_TE=np.empty((angles.shape[0],self.wl.shape[0]),dtype=float)
        T_TM=np.empty((angles.shape[0],self.wl.shape[0]),dtype=float)
        S=np.empty((2,2,self.wl.shape[0]),dtype=complex)
        S=np.empty((2,2,self.wl.shape[0]),dtype=complex)
        S_TM=np.empty((2,2,self.wl.shape[0]),dtype=complex)
        Ijk=np.empty((2,2,self.wl.shape[0]),dtype=complex)
        IjkTM=np.empty((2,2,self.wl.shape[0]),dtype=complex)
        Lj=np.empty((2,2,self.wl.shape[0]),dtype=complex)
        bj=np.empty(self.wl.shape[0],dtype=complex)
        
        df=pd.DataFrame(columns=['wavelength','angle','R_TE','R_TM','T_TE','T_TM'])
        dfn=pd.DataFrame(columns=['wavelength','angle','R_TE','R_TM','T_TE','T_TM'])
        dfn['wavelength']=self.wl

        for idx,theta in enumerate(angles):    
        
        # from air to first layer
        
            qj = self.nair*np.cos(theta)
            
            qk = np.sqrt(self.index[0]**2-(self.nair**2)*(np.sin(theta))**2)
            
            rjk = (qj-qk)/(qj+qk)
            tjk = 2*qj/(qj+qk)
            
            rjkTM = (-self.index[0]**2*qj+self.nair**2*qk)/(self.index[0]**2*qj+
                                                                self.nair**2*qk) 
            tjkTM = 2*self.index[0]*self.nair*qj/(self.index[0]**2*qj+self.nair**2*qk)
            
            S[0, 0, :] = 1/tjk
            S[0, 1, :] = rjk/tjk
            S[1, 0, :] = S[0, 1, :]
            S[1, 1, :] = S[0, 0, :]
            
            S_TM[0,0,:]=1/tjkTM;
            S_TM[0,1,:]=rjkTM/tjkTM
            S_TM[1,0,:]=S_TM[0,1,:]
            S_TM[1,1,:]=S_TM[0,0,:]
            
        # all other layers
        
            for k in range(self.numlayers):
                qj=np.sqrt(self.index[k]**2-(self.nair**2)*(np.sin(theta))**2)
                qk=np.sqrt(self.index[k+1]**2-(self.nair**2)*(np.sin(theta))**2)
                
                bj = 2*np.pi*qj/self.wl
                
                Lj[0,0,:]=np.exp(-1j*bj*self.thick[k])
                Lj[0,1,:]=0
                Lj[1,0,:]=0
                Lj[1,1,:]=np.exp(1j*bj*self.thick[k])
                
                rjk=(qj-qk)/(qj+qk)
                tjk=2*qj/(qj+qk)
            
                rjkTM=(-self.index[k+1]**2*qj+self.index[k]**2*qk)/((self.index[k+1]**2)*qj+(self.index[k]**2)*qk)
                tjkTM=2*self.index[k+1]*self.index[k]*qj/(self.index[k+1]**2*qj+self.index[k]**2*qk)
                
                Ijk[0,0,:]=1/tjk;
                Ijk[0,1,:]=rjk/tjk;
                Ijk[1,0,:]=Ijk[0,1,:]
                Ijk[1,1,:]=Ijk[0,0,:]
                
                IjkTM[0,0,:]=1/tjkTM;
                IjkTM[0,1,:]=rjkTM/tjkTM
                IjkTM[1,0,:]=IjkTM[0,1,:]
                IjkTM[1,1,:]=IjkTM[0,0,:]
                
                for kk in range(self.wl.shape[0]):
                    S[:,:,kk]=S[:,:,kk] @ Lj[:,:,kk] @ Ijk[:,:,kk]
                    S_TM[:,:,kk]=S_TM[:,:,kk] @ Lj[:,:,kk] @ IjkTM[:,:,kk]
                
                
            q0=np.sqrt(self.nair**2-(self.nair**2)*(np.sin(theta))**2)/self.nair
            qsubs=np.sqrt(self.nsubs**2-(self.nair**2)*(np.sin(theta))**2)/self.nsubs
              
            
            #R_TM[idx,:]=np.abs(S_TM[1,0,:]/S_TM[0,0,:])**2
            #R_TE[idx,:]=np.abs(S[1,0,:]/S[0,0,:])**2
            #T_TM[idx,:]=(qsubs/q0)*(self.nsubs/self.nair)*(np.abs(1/S_TM[0,0,:])**2)
            #T_TE[idx,:]=(qsubs/q0)*(self.nsubs/self.nair)*(np.abs(1/S[0,0,:])**2)

            dfn['angle']=180*theta/np.pi
           
            dfn['R_TM'] =np.abs(S_TM[1,0,:]/S_TM[0,0,:])**2
            dfn['R_TE'] =np.abs(S[1,0,:]/S[0,0,:])**2           
            dfn['T_TM'] =(qsubs/q0)*(self.nsubs/self.nair)*(np.abs(1/S_TM[0,0,:])**2)
            dfn['T_TE'] =(qsubs/q0)*(self.nsubs/self.nair)*(np.abs(1/S[0,0,:])**2)
            df=pd.concat([df,dfn])
        return df


# fig = go.Figure(go.Line(x=wl, y=R_TE.T))
# fig.update_layout(title_text='hello world')
# pio.write_html(fig, file='hello_world.html', auto_open=True)

# py.iplot(data, filename = 'basic-line')

#figure, ax = plt.subplots(1,2,figsize=(10,4))

#ax[0].set_title("TE Reflectivity") 
#ax[0].set_xlabel("Wavelength [nm]") 
#ax[0].set_ylabel("Reflectivity") 
#ax[0].plot(wl,R_TE.T) 

#ax[1].set_title("TM Reflectivity") 
#ax[1].set_xlabel("Wavelength [nm]") 
#ax[1].plot(wl,R_TM.T) 
#plt.show()
#plt.savefig('reflectivity.pdf',dpi=600)

# figure (1)
# plot(1239.85./lambda,R_TE(:,:));
# figure (2)
# plot(1239.85./lambda,T_TE(:,:));
# figure (3)
# plot(1239.85./lambda,1-T_TE(:,:)-R_TE(:,:));

# % figure (1)
# % [X,Y]=meshgrid(angles,lambda);
# % mesh(X,Y,R_TE');
# % view(0,-90)
# % figure (2)
# % mesh(X,Y,R_TM');
# % view(0,-90)


# %legend('15','20','25','30','35','40','45','50','55','60','65','70','75');