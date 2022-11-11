import numpy as np
from abc import ABC, abstractmethod

from scipy.spatial.distance import pdist, cdist, squareform

from time import time

class Kernel(ABC):
    def __init__(self):
        self._theta = None

    @abstractmethod
    def __call__(self):
        pass

    @abstractmethod
    def kernel(self):
        pass

    @abstractmethod
    def kernel_vector(self):
        pass

    @abstractmethod
    def kernel_jacobian(self):
        pass

    @abstractmethod
    def kernel_hyperparameter_gradient(self):
        pass

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        self._theta = theta
    
    def numerical_jacobian(self,x,y, dx=1.e-5):
        if np.ndim(y) == 1:
            y = y.reshape((1,-1))
        nx = len(x)
        ny = y.shape[0]
        f_jac = np.zeros((ny,nx))
        for i in range(nx):
            x_up = np.copy(x)
            x_down = np.copy(x)
            x_up[i] += 0.5*dx
            x_down[i] -= 0.5*dx
            
            f_up = self.kernel(x_up,y)
            f_down = self.kernel(x_down,y)

            f_jac[:,i] = (f_up - f_down)/dx
        return f_jac

    def numerical_hyperparameter_gradient(self,X, dx=1.e-5):
        """Calculates the numerical derivative of the kernel with respect to the
        log transformed hyperparameters.
        """
        N_data = X.shape[0]
        theta = np.copy(self.theta)
        N_hyper = len(theta)
        dK_dTheta = np.zeros((N_hyper, N_data, N_data))
        for i in range(N_hyper):
            theta_up = np.copy(theta)
            theta_down = np.copy(theta)
            theta_up[i] += 0.5*dx
            theta_down[i] -= 0.5*dx
            
            self.theta = theta_up
            K_up = self(X, eval_gradient=False)
            self.theta = theta_down
            K_down = self(X, eval_gradient=False)

            dK_dTheta[i,:,:] = (K_up - K_down)/dx
        self.theta = theta
        return dK_dTheta


class GaussKernel(Kernel):
    def __init__(self, amplitude=100.0, amplitude_bounds=(1e0, 1e5),
                 length_scale=10.0, length_scale_bounds=(1e-1, 1e3),
                 noise=1e-5, noise_bounds=None,
                 eta=1, eta_bounds=(0.1,10),
                 Nsplit_eta=None):
        self.amplitude = amplitude
        self.length_scale = length_scale
        self.noise = noise
        self.eta = eta
        self.Nsplit_eta = Nsplit_eta

        self.set_theta_bounds(amplitude_bnd=amplitude_bounds,
                              l_bnd=length_scale_bounds,
                              noise_bnd=(noise, noise),
                              eta_bnd=eta_bounds)

    def set_theta_bounds(self, amplitude_bnd=None, l_bnd=None, noise_bnd=None, eta_bnd=None):
        if amplitude_bnd is not None:
            self.amplitude_bounds = amplitude_bnd
        if l_bnd is not None:
            self.length_scale_bounds = l_bnd
        if noise_bnd is not None:
            self.noise_bounds = noise_bnd
        if self.Nsplit_eta is not None:
            if eta_bnd is not None:
                self.eta_bounds = eta_bnd
        else:
            self.eta_bounds = (self.eta,self.eta)

        self.theta_bounds = np.log(np.array([self.amplitude_bounds, self.length_scale_bounds, self.noise_bounds, self.eta_bounds]))

    def __call__(self, X, eval_gradient=False):
        K = self.kernel(X, with_noise=True)
        if eval_gradient:
            K_gradient = self.kernel_hyperparameter_gradient(X)
            return K, K_gradient
        else:
            return K

    def kernel(self, X, Y=None, with_noise=False):
        if with_noise:
            assert Y is None
        if np.ndim(X) == 1:
            X = X.reshape((1,-1))
        if Y is None:
            Y = X
        elif np.ndim(Y) == 1:
            Y = Y.reshape((1,-1))
        X = self.apply_eta(X)
        Y = self.apply_eta(Y)
        d = cdist(X / self.length_scale,
                  Y / self.length_scale, metric='sqeuclidean')
        if with_noise:
            K = self.amplitude * (np.exp(-0.5 * d) + self.noise*np.eye(X.shape[0]))
        else:
            K = self.amplitude * np.exp(-0.5 * d)
        return K

    def kernel_value(self, x,y):
        K = self.kernel(x,y)
        return np.asscalar(K)
    
    def kernel_vector(self, x,Y):
        K = self.kernel(x,Y).reshape(-1)
        return K

    def kernel_jacobian(self, X,Y, trim_shape=False):
        """ Jacobian of the kernel with respect to X

        Output has shape N_X x N_Y x N_features
        """
        X = self.apply_eta(X)
        Y = self.apply_eta(Y)
        if np.ndim(X) == 1:
            X = X.reshape((1,-1))
        if np.ndim(Y) == 1:
            Y = Y.reshape((1,-1))
        d = cdist(X / self.length_scale,
                  Y / self.length_scale, metric='sqeuclidean')
        dK_dd = -self.amplitude * 1/(2*self.length_scale**2) * np.exp(-0.5 * d)
        X = self.apply_eta(X)
        Y = self.apply_eta(Y)

        N_X, Nf = X.shape
        N_Y = Y.shape[0]

        X_rep = np.tile(X, (N_Y,1,1)).swapaxes(0,1)
        Y_rep = np.tile(Y, (N_X,1,1))

        dd_dX = 2*(X_rep-Y_rep)  # shape: (N_X, N_Y, Nf)

        dK_dX = dK_dd.reshape(N_X,N_Y,1) * dd_dX
        if trim_shape and N_X == 1:
            return dK_dX.reshape(-1,Nf)
        else:
            return dK_dX

    def kernel_hessian_old(self, X,Y):
        """ Jacobian of the kernel with respect to X

        Output has shape N_X x N_Y x N_features x N_features
        """
        t0 = time()
        X = self.apply_eta(X)
        Y = self.apply_eta(Y)
        if np.ndim(X) == 1:
            X = X.reshape((1,-1))
        if np.ndim(Y) == 1:
            Y = Y.reshape((1,-1))
        d = cdist(X / self.length_scale,
                  Y / self.length_scale, metric='sqeuclidean')
        u = -1/(2*self.length_scale**2)
        K = self.amplitude * np.exp(-0.5 * d)
        X = self.apply_eta(X)
        Y = self.apply_eta(Y)

        N_X, Nf = X.shape
        N_Y = Y.shape[0]

        t1 = time()

        X_rep = np.tile(X, (N_Y,1,1)).swapaxes(0,1)
        Y_rep = np.tile(Y, (N_X,1,1))

        t2 = time()

        dd_dX = 2*(X_rep-Y_rep)  # shape: (N_X, N_Y, Nf)
        dd_dY = -dd_dX

        t3 = time()
        dd_dX_dd_dY = np.einsum('nmi,nmj->nmij',dd_dX, dd_dY)  # shape: (N_X, N_Y, Nf, Nf)
        t4 = time()
        if self.Nsplit_eta is not None:
            diag = np.diag(self.Nsplit_eta*[1]+(Nf-self.Nsplit_eta)*[self.eta**2])
        else:
            diag = np.identity(Nf)
        t5 = time()
        d2d_dXdY = np.tile(-2*diag, (N_X, N_Y,1,1))  # shape: (N_X, N_Y, Nf, Nf)
        t6 = time()
        hess = u*K.reshape(N_X,N_Y,1,1) * (u*dd_dX_dd_dY + d2d_dXdY)
        #hess = u*K.reshape(N_X,N_Y,1,1) * (d2d_dXdY)
        #hess = (d2d_dXdY)
        #hess = u*K.reshape(N_X,N_Y,1,1) * (u*dd_dX_dd_dY)
        #hess = dd_dX_dd_dY
        t7 = time()
        #print(f'i: {t1-t0:.1e}, rep: {t2-t1:.1e}, dd: {t3-t2:.1e}, ein: {t4-t3:.1e}, diag: {t5-t4:.1e}, d2d: {t6-t5:.1e}, hess: {t7-t6:.1e}, tot: {t7-t0:.1e}')

        return hess

    def kernel_hessian(self, X,Y, dX_dr, dY_dr, with_noise=False):
        """ Jacobian of the kernel with respect to X

        Output has shape N_X x N_Y x N_features x N_features
        """
        X = self.apply_eta(X)
        Y = self.apply_eta(Y)
        if np.ndim(X) == 1:
            X = X.reshape((1,-1))
        if np.ndim(Y) == 1:
            Y = Y.reshape((1,-1))
        d = cdist(X / self.length_scale,
                  Y / self.length_scale, metric='sqeuclidean')
        u = -1/(2*self.length_scale**2)
        K = self.amplitude * np.exp(-0.5 * d)
        dX_dr = self.apply_eta(dX_dr)
        dY_dr = self.apply_eta(dY_dr)

        N_X, N_dX, Nf = dX_dr.shape
        N_Y, N_dY, _ = dY_dr.shape

        # Evaluate dd_drX
        X_dX_dr = np.einsum('ijk,ink->ijn',X.reshape(N_X,1,Nf), dX_dr)  # shape: (N_X x 1 x N_dX)
        Y_dX_dr = np.einsum('ijk,ink->ijn',Y.reshape(1,N_Y,Nf), dX_dr)  # shape: (1 x N_Y x N_dX)
        dd_drX = 2*(X_dX_dr-Y_dX_dr)  # shape: (N_X x N_Y x N_dX)

        # Evaluate dd_drY
        X_dY_dr = np.einsum('ijk,jnk->ijn',X.reshape(N_X,1,Nf), dY_dr)  # shape: (N_X x 1 x N_dY)
        Y_dY_dr = np.einsum('ijk,jnk->ijn',Y.reshape(1,N_Y,Nf), dY_dr)  # shape: (1 x N_Y x N_dY)
        dd_drY = 2*(Y_dY_dr-X_dY_dr)  # shape: (N_X x N_Y x N_dY)

        dd_drX_dd_drY = np.einsum('ijn,ijm->injm',dd_drX, dd_drY)  # shape: (N_X, N_Y, N_dX, N_dY)
        d2d_drXdrY = -2*np.einsum('imk,jnk->imjn', dX_dr, dY_dr)  # shape: (N_X, N_Y, N_dX, N_dY)

        hess = u*K.reshape(N_X,1,N_Y,1) * (u*dd_drX_dd_drY + d2d_drXdrY)
        hess = hess.reshape(N_X*N_dX,N_Y*N_dY)
        if with_noise:
            hess += self.amplitude*self.noise*np.eye(hess.shape[0])  # Add regularization equal to that on the function values.
        return hess

    @property
    def theta(self):
        """Returns the log-transformed hyperparameters of the kernel.
        """
        self._theta = np.array([self.amplitude, self.length_scale, self.noise, self.eta]) 
        return np.log(self._theta)

    @theta.setter
    def theta(self, theta):
        """Sets the hyperparameters of the kernel.

        theta: log-transformed hyperparameters
        """
        self._theta = np.exp(theta)
        self.amplitude = self._theta[0]
        self.length_scale = self._theta[1]
        self.noise = self._theta[2]
        self.eta = self._theta[3]

    def apply_eta(self, X):
        Xeta = np.copy(X)
        if self.Nsplit_eta is not None:
            if np.ndim(X) == 1:
                Xeta[self.Nsplit_eta:] *= self.eta
            elif np.ndim(X) == 2:
                Xeta[:,self.Nsplit_eta:] *= self.eta
            elif np.ndim(X) == 3:
                Xeta[:,:,self.Nsplit_eta:] *= self.eta
        return Xeta

    def dK_da(self, X):
        d = cdist(X / self.length_scale,
                  X / self.length_scale, metric='sqeuclidean')
        dK_da = self.amplitude * (np.exp(-0.5 * d) + self.noise*np.eye(X.shape[0]))
        return dK_da
        
    def dK_dl(self, X):
        d = cdist(X / self.length_scale,
                  X / self.length_scale, metric='sqeuclidean')
        dK_dl = self.amplitude * d * np.exp(-0.5 * d)
        return dK_dl

    def dK_dn(self, X):
        dK_dn = self.amplitude * self.noise * np.eye(X.shape[0])
        return dK_dn

    def dK_deta(self, X, dx=1e-5):
        N_data = X.shape[0]
        theta = np.copy(self.theta)
        dK_deta = np.zeros((N_data, N_data))

        theta_up = np.copy(theta)
        theta_down = np.copy(theta)
        theta_up[-1] += 0.5*dx
        theta_down[-1] -= 0.5*dx
        
        self.theta = theta_up
        K_up = self(X, eval_gradient=False)
        self.theta = theta_down
        K_down = self(X, eval_gradient=False)
        dK_dTheta = (K_up - K_down)/dx

        self.theta = theta
        return dK_dTheta

    def kernel_hyperparameter_gradient(self, X):
        """Calculates the derivative of the kernel with respect to the
        log transformed hyperparameters.
        """
        dK_deta = self.dK_deta(X)
        X = self.apply_eta(X)
        return np.array([self.dK_da(X), self.dK_dl(X), self.dK_dn(X), dK_deta])

        

class DoubleGaussKernel(Kernel):
    def __init__(self, amplitude=100., amplitude_bounds=(1e0,1e5),
                 length_scale1=10.0, length_scale1_bounds=(1e-1, 1e3),
                 length_scale2=10.0, length_scale2_bounds=(1e-1, 1e3),
                 weight=0.01, weight_bounds=None,
                 noise=1e-5, noise_bounds=None,
                 eta=1, eta_bounds=(0.1,10),
                 Nsplit_eta=None,
                 dynamic_noise=True,
                 noise_hess=None):
        self.amplitude = amplitude
        self.length_scale1 = length_scale1
        self.length_scale2 = length_scale2
        self.weight = weight
        self.noise = noise
        self.eta = eta
        self.Nsplit_eta = Nsplit_eta
        self.dynamic_noise = dynamic_noise
        if noise_bounds is None:
            noise_bounds = (noise, noise)
        if weight_bounds is None:
            weight_bounds = (weight, weight)
        if noise_hess is None:
            self.noise_hess = noise
        else:
            self.noise_hess = noise_hess

        self.set_theta_bounds(amplitude_bnd=amplitude_bounds,
                              l1_bnd=length_scale1_bounds,
                              l2_bnd=length_scale2_bounds,
                              w_bnd=weight_bounds,
                              noise_bnd=noise_bounds,
                              eta_bnd=eta_bounds)

    def set_theta_bounds(self, amplitude_bnd=None, l1_bnd=None, l2_bnd=None, w_bnd=None, noise_bnd=None, eta_bnd=None):
        if amplitude_bnd is not None:
            self.amplitude_bounds = amplitude_bnd
        if l1_bnd is not None:
            self.length_scale1_bounds = l1_bnd
        if l2_bnd is not None:
            self.length_scale2_bounds = l2_bnd
        if w_bnd is not None:
            self.weight_bounds = w_bnd
        if noise_bnd is not None:
            self.noise_bounds = noise_bnd
        if self.Nsplit_eta is not None:
            if eta_bnd is not None:
                self.eta_bounds = eta_bnd
        else:
            self.eta_bounds = (self.eta,self.eta)

        self.theta_bounds = np.log(np.array([self.amplitude_bounds, self.length_scale1_bounds, self.length_scale2_bounds, self.weight_bounds, self.noise_bounds, self.eta_bounds]))

    def __call__(self, X, eval_gradient=False, reg_scaling=None):
        if reg_scaling is not None:
            reg_add = np.diag(reg_scaling)
        else:
            reg_add = 0
        K = self.kernel(X, with_noise=True, reg_scaling=reg_add)
        if eval_gradient:
            K_gradient = self.kernel_hyperparameter_gradient(X, reg_scaling=reg_add)
            return K, K_gradient
        else:
            return K

    def kernel(self, X, Y=None, with_noise=False, reg_scaling=None):
        if with_noise:
            assert Y is None
        if np.ndim(X) == 1:
            X = X.reshape((1,-1))
        if Y is None:
            Y = X
        elif np.ndim(Y) == 1:
            Y = Y.reshape((1,-1))
        X = self.apply_eta(X)
        Y = self.apply_eta(Y)
        d1 = cdist(X / self.length_scale1,
                  Y / self.length_scale1, metric='sqeuclidean')
        d2 = cdist(X / self.length_scale2,
                  Y / self.length_scale2, metric='sqeuclidean')
        if with_noise:
            if self.dynamic_noise:
                K = self.amplitude * (np.exp(-0.5 * d1) + self.weight*np.exp(-0.5 * d2) + self.noise*np.eye(X.shape[0])) + reg_scaling
            else:
                K = self.amplitude * (np.exp(-0.5 * d1) + self.weight*np.exp(-0.5 * d2)) + self.noise*np.eye(X.shape[0]) + reg_scaling
        else:
            K = self.amplitude * (np.exp(-0.5 * d1) + self.weight*np.exp(-0.5 * d2))
        return K

    def kernel_value(self, x,y, reg_scaling=None):
        K = self.kernel(x,y, reg_scaling=reg_scaling)
        return np.asscalar(K)
    
    def kernel_vector(self, x,Y):
        K = self.kernel(x,Y).reshape(-1)
        return K

    def kernel_jacobian(self, X,Y, trim_shape=False):
        """ Jacobian of the kernel with respect to X

        Output has shape N_X x N_Y x N_features
        """
        X = self.apply_eta(X)
        Y = self.apply_eta(Y)
        if np.ndim(X) == 1:
            X = X.reshape((1,-1))
        if np.ndim(Y) == 1:
            Y = Y.reshape((1,-1))
        d1 = cdist(X / self.length_scale1,
                   Y / self.length_scale1, metric='sqeuclidean')
        d2 = cdist(X / self.length_scale2,
                   Y / self.length_scale2, metric='sqeuclidean')
        dK1_dd1 = -1/(2*self.length_scale1**2) * np.exp(-0.5 * d1)
        dK2_dd2 = -1/(2*self.length_scale2**2) * np.exp(-0.5 * d2)
        dK_dd = self.amplitude * (dK1_dd1 + self.weight*dK2_dd2)
        X = self.apply_eta(X)
        Y = self.apply_eta(Y)

        N_X, Nf = X.shape
        N_Y = Y.shape[0]

        X_rep = np.tile(X, (N_Y,1,1)).swapaxes(0,1)
        Y_rep = np.tile(Y, (N_X,1,1))

        dd_dX = 2*(X_rep-Y_rep)  # shape: (N_X, N_Y, Nf)

        dK_dX = dK_dd.reshape(N_X,N_Y,1) * dd_dX
        if trim_shape and N_X == 1:
            return dK_dX.reshape(-1,Nf)
        else:
            return dK_dX

    def kernel_hessian(self, X,Y, dX_dr, dY_dr, with_noise=False):
        """ Jacobian of the kernel with respect to X

        Output has shape N_X x N_Y x N_features x N_features
        """
        X = self.apply_eta(X)
        Y = self.apply_eta(Y)
        if np.ndim(X) == 1:
            X = X.reshape((1,-1))
        if np.ndim(Y) == 1:
            Y = Y.reshape((1,-1))
        d1 = cdist(X / self.length_scale1,
                   Y / self.length_scale1, metric='sqeuclidean')
        d2 = cdist(X / self.length_scale2,
                   Y / self.length_scale2, metric='sqeuclidean')
        u1 = -1/(2*self.length_scale1**2)
        u2 = -1/(2*self.length_scale2**2)
        K1 = np.exp(-0.5 * d1)
        K2 = self.weight*np.exp(-0.5 * d2)
        dX_dr = self.apply_eta(dX_dr)
        dY_dr = self.apply_eta(dY_dr)

        N_X, N_dX, Nf = dX_dr.shape
        N_Y, N_dY, _ = dY_dr.shape

        # Evaluate dd_drX
        X_dX_dr = np.einsum('ijk,ink->ijn',X.reshape(N_X,1,Nf), dX_dr)  # shape: (N_X x 1 x N_dX)
        Y_dX_dr = np.einsum('ijk,ink->ijn',Y.reshape(1,N_Y,Nf), dX_dr)  # shape: (1 x N_Y x N_dX)
        dd_drX = 2*(X_dX_dr-Y_dX_dr)  # shape: (N_X x N_Y x N_dX)

        # Evaluate dd_drY
        X_dY_dr = np.einsum('ijk,jnk->ijn',X.reshape(N_X,1,Nf), dY_dr)  # shape: (N_X x 1 x N_dY)
        Y_dY_dr = np.einsum('ijk,jnk->ijn',Y.reshape(1,N_Y,Nf), dY_dr)  # shape: (1 x N_Y x N_dY)
        dd_drY = 2*(Y_dY_dr-X_dY_dr)  # shape: (N_X x N_Y x N_dY)

        dd_drX_dd_drY = np.einsum('ijn,ijm->injm',dd_drX, dd_drY)  # shape: (N_X, N_Y, N_dX, N_dY)
        d2d_drXdrY = -2*np.einsum('imk,jnk->imjn', dX_dr, dY_dr)  # shape: (N_X, N_Y, N_dX, N_dY)

        K1 = K1.reshape(N_X,1,N_Y,1)
        K2 = K2.reshape(N_X,1,N_Y,1)
        hess = self.amplitude * ( (u1**2*K1 + u2**2*K2)*dd_drX_dd_drY + (u1*K1 + u2*K2)*d2d_drXdrY)
        hess = hess.reshape(N_X*N_dX,N_Y*N_dY)
        if with_noise:
            if self.dynamic_noise:
                hess += self.amplitude*self.noise*np.eye(hess.shape[0])  # Add regularization equal to that on the function values.
            else:
                hess += self.noise_hess*np.eye(hess.shape[0])
        return hess

    @property
    def theta(self):
        """Returns the log-transformed hyperparameters of the kernel.
        """
        self._theta = np.array([self.amplitude, self.length_scale1, self.length_scale2, self.weight, self.noise, self.eta])
        return np.log(self._theta)

    @theta.setter
    def theta(self, theta):
        """Sets the hyperparameters of the kernel.

        theta: log-transformed hyperparameters
        """
        self._theta = np.exp(theta)
        self.amplitude = self._theta[0]
        self.length_scale1 = self._theta[1]
        self.length_scale2 = self._theta[2]
        self.weight = self._theta[3]
        self.noise = self._theta[4]
        self.eta = self._theta[5]

    def apply_eta(self, X):
        Xeta = np.copy(X)
        if self.Nsplit_eta is not None:
            if np.ndim(X) == 1:
                Xeta[self.Nsplit_eta:] *= self.eta
            else:
                Xeta[:,self.Nsplit_eta:] *= self.eta
        return Xeta

    def dK_da(self, X):
        d1 = cdist(X / self.length_scale1,
                   X / self.length_scale1, metric='sqeuclidean')
        d2 = cdist(X / self.length_scale2,
                   X / self.length_scale2, metric='sqeuclidean')
        if self.dynamic_noise:
            dK_da = self.amplitude * (np.exp(-0.5 * d1) + self.weight*np.exp(-0.5 * d2) + self.noise*np.eye(X.shape[0]))
        else:
            dK_da = self.amplitude * (np.exp(-0.5 * d1) + self.weight*np.exp(-0.5 * d2))
        return dK_da
        
    def dK_dl1(self, X):
        d1 = cdist(X / self.length_scale1,
                   X / self.length_scale1, metric='sqeuclidean')
        dK_dl1 = self.amplitude*d1 * np.exp(-0.5 * d1)
        return dK_dl1

    def dK_dl2(self, X):
        d2 = cdist(X / self.length_scale2,
                   X / self.length_scale2, metric='sqeuclidean')
        dK_dl2 = self.amplitude*self.weight*d2 * np.exp(-0.5 * d2)
        return dK_dl2

    def dK_dw(self, X):
        d2 = cdist(X / self.length_scale2,
                   X / self.length_scale2, metric='sqeuclidean')
        dK_dl2 = self.amplitude*self.weight*np.exp(-0.5 * d2)
        return dK_dl2

    def dK_dn(self, X):
        if self.dynamic_noise:
            dK_dn = self.amplitude * self.noise * np.eye(X.shape[0])
        else:
            dK_dn = self.noise * np.eye(X.shape[0])
        return dK_dn

    def dK_deta(self, X, dx=1e-5, reg_scaling=None):
        N_data = X.shape[0]
        theta = np.copy(self.theta)
        dK_deta = np.zeros((N_data, N_data))

        theta_up = np.copy(theta)
        theta_down = np.copy(theta)
        theta_up[-1] += 0.5*dx
        theta_down[-1] -= 0.5*dx
        
        self.theta = theta_up
        K_up = self.kernel(X, with_noise=True, reg_scaling=reg_scaling)
        self.theta = theta_down
        K_down = self.kernel(X, with_noise=True, reg_scaling=reg_scaling)
        dK_dTheta = (K_up - K_down)/dx

        self.theta = theta
        return dK_dTheta

    def kernel_hyperparameter_gradient(self, X, reg_scaling=None):
        """Calculates the derivative of the kernel with respect to the
        log transformed hyperparameters.
        """
        dK_deta = self.dK_deta(X, reg_scaling=reg_scaling)
        X = self.apply_eta(X)
        return np.array([self.dK_da(X), self.dK_dl1(X), self.dK_dl2(X), self.dK_dw(X), self.dK_dn(X), dK_deta])


class DoubleGaussKernel2(Kernel):
    def __init__(self, amplitude=100., amplitude_bounds=(1e0,1e5),
                 length_scale=10.0, length_scale_bounds=(1e-1, 1e3),
                 length_scale_frac=0.5, length_scale_frac_bounds=(0.16, 1),  # (np.exp(0.166), np.exp(1)),
                 weight=0.01, weight_bounds=None,
                 noise=1e-5, noise_bounds=None,
                 eta=1, eta_bounds=(0.1,10),
                 Nsplit_eta=None,
                 dynamic_noise=True,
                 noise_hess=None):
        self.amplitude = amplitude
        self.length_scale = length_scale
        self.length_scale_frac = length_scale_frac
        self.weight = weight
        self.noise = noise
        self.eta = eta
        self.Nsplit_eta = Nsplit_eta
        self.dynamic_noise = dynamic_noise
        if noise_bounds is None:
            noise_bounds = (noise, noise)
        if weight_bounds is None:
            weight_bounds = (weight, weight)
        if noise_hess is None:
            self.noise_hess = noise
        else:
            self.noise_hess = noise_hess

        self.set_theta_bounds(amplitude_bnd=amplitude_bounds,
                              l_bnd=length_scale_bounds,
                              l_frac_bnd=length_scale_frac_bounds,
                              w_bnd=weight_bounds,
                              noise_bnd=noise_bounds,
                              eta_bnd=eta_bounds)

    def set_theta_bounds(self, amplitude_bnd=None, l_bnd=None, l_frac_bnd=None, w_bnd=None, noise_bnd=None, eta_bnd=None):
        if amplitude_bnd is not None:
            self.amplitude_bounds = amplitude_bnd
        if l_bnd is not None:
            self.length_scale_bounds = l_bnd
        if l_frac_bnd is not None:
            self.length_scale_frac_bounds = l_frac_bnd
        if w_bnd is not None:
            self.weight_bounds = w_bnd
        if noise_bnd is not None:
            self.noise_bounds = noise_bnd
        if self.Nsplit_eta is not None:
            if eta_bnd is not None:
                self.eta_bounds = eta_bnd
        else:
            self.eta_bounds = (self.eta,self.eta)

        self.theta_bounds = np.log(np.array([self.amplitude_bounds, self.length_scale_bounds, self.length_scale_frac_bounds, self.weight_bounds, self.noise_bounds, self.eta_bounds]))

    def __call__(self, X, eval_gradient=False, reg_scaling=None):
        if reg_scaling is not None:
            reg_add = np.diag(reg_scaling)
        else:
            reg_add = 0
        K = self.kernel(X, with_noise=True, reg_scaling=reg_add)
        if eval_gradient:
            K_gradient = self.kernel_hyperparameter_gradient(X, reg_scaling=reg_add)
            return K, K_gradient
        else:
            return K

    def kernel(self, X, Y=None, with_noise=False, reg_scaling=None):
        if with_noise:
            assert Y is None
        if np.ndim(X) == 1:
            X = X.reshape((1,-1))
        if Y is None:
            Y = X
        elif np.ndim(Y) == 1:
            Y = Y.reshape((1,-1))
        X = self.apply_eta(X)
        Y = self.apply_eta(Y)
        l1 = self.length_scale
        l2 = self.length_scale*self.length_scale_frac
        d1 = cdist(X / l1,
                  Y / l1, metric='sqeuclidean')
        d2 = cdist(X / l2,
                  Y / l2, metric='sqeuclidean')
        if with_noise:
            if self.dynamic_noise:
                K = self.amplitude * (np.exp(-0.5 * d1) + self.weight*np.exp(-0.5 * d2) + self.noise*np.eye(X.shape[0])) + reg_scaling
            else:
                K = self.amplitude * (np.exp(-0.5 * d1) + self.weight*np.exp(-0.5 * d2)) + self.noise*np.eye(X.shape[0]) + reg_scaling
        else:
            K = self.amplitude * (np.exp(-0.5 * d1) + self.weight*np.exp(-0.5 * d2))
        return K

    def kernel_value(self, x,y, reg_scaling=None):
        K = self.kernel(x,y, reg_scaling=reg_scaling)
        return np.asscalar(K)
    
    def kernel_vector(self, x,Y):
        K = self.kernel(x,Y).reshape(-1)
        return K

    def kernel_jacobian(self, X,Y, trim_shape=False):
        """ Jacobian of the kernel with respect to X

        Output has shape N_X x N_Y x N_features
        """
        X = self.apply_eta(X)
        Y = self.apply_eta(Y)
        if np.ndim(X) == 1:
            X = X.reshape((1,-1))
        if np.ndim(Y) == 1:
            Y = Y.reshape((1,-1))
        l1 = self.length_scale
        l2 = self.length_scale*self.length_scale_frac
        d1 = cdist(X / l1,
                  Y / l1, metric='sqeuclidean')
        d2 = cdist(X / l2,
                  Y / l2, metric='sqeuclidean')
        dK1_dd1 = -1/(2*l1**2) * np.exp(-0.5 * d1)
        dK2_dd2 = -1/(2*l2**2) * np.exp(-0.5 * d2)
        dK_dd = self.amplitude * (dK1_dd1 + self.weight*dK2_dd2)
        X = self.apply_eta(X)
        Y = self.apply_eta(Y)

        N_X, Nf = X.shape
        N_Y = Y.shape[0]

        X_rep = np.tile(X, (N_Y,1,1)).swapaxes(0,1)
        Y_rep = np.tile(Y, (N_X,1,1))

        dd_dX = 2*(X_rep-Y_rep)  # shape: (N_X, N_Y, Nf)

        dK_dX = dK_dd.reshape(N_X,N_Y,1) * dd_dX
        if trim_shape and N_X == 1:
            return dK_dX.reshape(-1,Nf)
        else:
            return dK_dX

    def kernel_hessian(self, X,Y, dX_dr, dY_dr, with_noise=False):
        """ Jacobian of the kernel with respect to X

        Output has shape N_X x N_Y x N_features x N_features
        """
        X = self.apply_eta(X)
        Y = self.apply_eta(Y)
        if np.ndim(X) == 1:
            X = X.reshape((1,-1))
        if np.ndim(Y) == 1:
            Y = Y.reshape((1,-1))
        l1 = self.length_scale
        l2 = self.length_scale*self.length_scale_frac
        d1 = cdist(X / l1,
                  Y / l1, metric='sqeuclidean')
        d2 = cdist(X / l2,
                  Y / l2, metric='sqeuclidean')
        u1 = -1/(2*l1**2)
        u2 = -1/(2*l2**2)
        K1 = np.exp(-0.5 * d1)
        K2 = self.weight*np.exp(-0.5 * d2)
        dX_dr = self.apply_eta(dX_dr)
        dY_dr = self.apply_eta(dY_dr)

        N_X, N_dX, Nf = dX_dr.shape
        N_Y, N_dY, _ = dY_dr.shape

        # Evaluate dd_drX
        X_dX_dr = np.einsum('ijk,ink->ijn',X.reshape(N_X,1,Nf), dX_dr)  # shape: (N_X x 1 x N_dX)
        Y_dX_dr = np.einsum('ijk,ink->ijn',Y.reshape(1,N_Y,Nf), dX_dr)  # shape: (1 x N_Y x N_dX)
        dd_drX = 2*(X_dX_dr-Y_dX_dr)  # shape: (N_X x N_Y x N_dX)

        # Evaluate dd_drY
        X_dY_dr = np.einsum('ijk,jnk->ijn',X.reshape(N_X,1,Nf), dY_dr)  # shape: (N_X x 1 x N_dY)
        Y_dY_dr = np.einsum('ijk,jnk->ijn',Y.reshape(1,N_Y,Nf), dY_dr)  # shape: (1 x N_Y x N_dY)
        dd_drY = 2*(Y_dY_dr-X_dY_dr)  # shape: (N_X x N_Y x N_dY)

        dd_drX_dd_drY = np.einsum('ijn,ijm->injm',dd_drX, dd_drY)  # shape: (N_X, N_Y, N_dX, N_dY)
        d2d_drXdrY = -2*np.einsum('imk,jnk->imjn', dX_dr, dY_dr)  # shape: (N_X, N_Y, N_dX, N_dY)

        K1 = K1.reshape(N_X,1,N_Y,1)
        K2 = K2.reshape(N_X,1,N_Y,1)
        hess = self.amplitude * ( (u1**2*K1 + u2**2*K2)*dd_drX_dd_drY + (u1*K1 + u2*K2)*d2d_drXdrY)
        hess = hess.reshape(N_X*N_dX,N_Y*N_dY)
        if with_noise:
            if self.dynamic_noise:
                hess += self.amplitude*self.noise*np.eye(hess.shape[0])  # Add regularization equal to that on the function values.
            else:
                hess += self.noise_hess*np.eye(hess.shape[0])
        return hess

    @property
    def theta(self):
        """Returns the log-transformed hyperparameters of the kernel.
        """
        self._theta = np.array([self.amplitude, self.length_scale, self.length_scale_frac, self.weight, self.noise, self.eta])
        return np.log(self._theta)

    @theta.setter
    def theta(self, theta):
        """Sets the hyperparameters of the kernel.

        theta: log-transformed hyperparameters
        """
        self._theta = np.exp(theta)
        self.amplitude = self._theta[0]
        self.length_scale = self._theta[1]
        self.length_scale_frac = self._theta[2]
        self.weight = self._theta[3]
        self.noise = self._theta[4]
        self.eta = self._theta[5]

    def apply_eta(self, X):
        Xeta = np.copy(X)
        if self.Nsplit_eta is not None:
            if np.ndim(X) == 1:
                Xeta[:self.Nsplit_eta] *= self.eta
            else:
                Xeta[:,:self.Nsplit_eta] *= self.eta
        return Xeta

    def dK_da(self, X):
        d1 = cdist(X / self.length_scale,
                   X / self.length_scale, metric='sqeuclidean')
        d2 = cdist(X / self.length_scale*self.length_scale_frac,
                   X / self.length_scale*self.length_scale_frac, metric='sqeuclidean')
        if self.dynamic_noise:
            dK_da = self.amplitude * (np.exp(-0.5 * d1) + self.weight*np.exp(-0.5 * d2) + self.noise*np.eye(X.shape[0]))
        else:
            dK_da = self.amplitude * (np.exp(-0.5 * d1) + self.weight*np.exp(-0.5 * d2))
        return dK_da
        
    def dK_dl(self, X):
        l1 = self.length_scale
        l2 = self.length_scale*self.length_scale_frac
        d = cdist(X / l1,
                  X / l1, metric='sqeuclidean')
        d2 = cdist(X / l2,
                   X / l2, metric='sqeuclidean')
        #d2 = d / self.length_scale_frac**2
        dK_dl = self.amplitude * ( d*np.exp(-0.5 * d) + d2*np.exp(-0.5 * d2) )
        return dK_dl

    def dK_dl_frac(self, X):
        l2 = self.length_scale*self.length_scale_frac
        d = cdist(X / l2,
                  X / l2, metric='sqeuclidean')
        dK_dl_frac = self.amplitude*self.weight*d * np.exp(-0.5 * d)
        return dK_dl_frac

    def dK_dw(self, X):
        l2 = self.length_scale*self.length_scale_frac
        d2 = cdist(X / l2,
                   X / l2, metric='sqeuclidean')
        dK_dl2 = self.amplitude*self.weight*np.exp(-0.5 * d2)
        return dK_dl2

    def dK_dn(self, X):
        if self.dynamic_noise:
            dK_dn = self.amplitude * self.noise * np.eye(X.shape[0])
        else:
            dK_dn = self.noise * np.eye(X.shape[0])
        return dK_dn

    def dK_deta(self, X, dx=1e-5, reg_scaling=None):
        N_data = X.shape[0]
        theta = np.copy(self.theta)
        dK_deta = np.zeros((N_data, N_data))

        theta_up = np.copy(theta)
        theta_down = np.copy(theta)
        theta_up[-1] += 0.5*dx
        theta_down[-1] -= 0.5*dx
        
        self.theta = theta_up
        K_up = self.kernel(X, with_noise=True, reg_scaling=reg_scaling)
        self.theta = theta_down
        K_down = self.kernel(X, with_noise=True, reg_scaling=reg_scaling)
        dK_dTheta = (K_up - K_down)/dx

        self.theta = theta
        return dK_dTheta

    def kernel_hyperparameter_gradient(self, X, reg_scaling=None):
        """Calculates the derivative of the kernel with respect to the
        log transformed hyperparameters.
        """
        dK_deta = self.dK_deta(X, reg_scaling=reg_scaling)
        X = self.apply_eta(X)
        return np.array([self.dK_da(X), self.dK_dl(X), self.dK_dl_frac(X), self.dK_dw(X), self.dK_dn(X), dK_deta])
