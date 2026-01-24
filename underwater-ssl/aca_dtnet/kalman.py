import torch

class KalmanCV_CVA:
    """
    1D range + velocity KF with continuous white-noise acceleration (CVA).
    State x=[r, v]. Measurement is range.
    """
    def __init__(self, dt=1.0, sigma_a=0.2, r_std=1.0, device="cpu", dtype=torch.float32):
        self.dt = float(dt)
        self.device, self.dtype = device, dtype

        self.F = torch.tensor([[1., dt],
                               [0., 1.]], device=device, dtype=dtype)            # (2,2)
        self.H = torch.tensor([[1., 0.]], device=device, dtype=dtype)              # (1,2)

        # CVA process noise
        dt3, dt2 = dt**3/3.0, dt**2/2.0
        Q = torch.tensor([[dt3, dt2],
                          [dt2, dt   ]], device=device, dtype=dtype)
        self.Q = (sigma_a**2) * Q                                                   # (2,2)

        self.R_fixed = torch.tensor([[r_std**2]], device=device, dtype=dtype)       # (1,1)
        self.I = torch.eye(2, device=device, dtype=dtype)

    @torch.no_grad()
    def run(self, z, R_t=None, x0=None, P0=None, smooth=False, v_clip=3.0):
        """
        z: (T,) or (T,1) network range predictions
        R_t: optional (T,) per-frame std-devs; if None, use fixed R
        Returns: filt (T,) or (filt, smooth) if smooth=True
        """
        z = z.view(-1, 1).to(self.device, self.dtype)
        T = z.shape[0]

        x = torch.zeros(2, device=self.device, dtype=self.dtype) if x0 is None else x0.to(self.device, self.dtype)
        if x0 is None:
            x[0] = z[0, 0]  # start from first measurement
        P = torch.diag(torch.tensor([100., 9.], device=self.device, dtype=self.dtype)) if P0 is None else P0.to(self.device, self.dtype)

        xs, Ps, x_preds, P_preds = [], [], [], []
        for t in range(T):
            # Predict
            x_pred = self.F @ x
            P_pred = self.F @ P @ self.F.T + self.Q

            # Measurement covariance
            if R_t is None:
                R = self.R_fixed
            else:
                rstd = torch.clamp(R_t[t].reshape(1,1).to(self.device, self.dtype), 1e-4, 10.0)
                R = rstd * rstd

            # Update
            y = z[t:t+1] - (self.H @ x_pred)            # innovation
            S = self.H @ P_pred @ self.H.T + R
            K = (P_pred @ self.H.T) @ torch.linalg.inv(S)

            x = x_pred + (K @ y).squeeze(1)
            P = (self.I - K @ self.H) @ P_pred

            # Physical constraints
            if v_clip is not None:
                x[1] = torch.clamp(x[1], -v_clip, v_clip)
            x[0] = torch.clamp(x[0], min=0.0)

            xs.append(x.clone()); Ps.append(P.clone())
            x_preds.append(x_pred.clone()); P_preds.append(P_pred.clone())

        xs = torch.stack(xs)             # (T,2)
        filt = xs[:, 0]                  # filtered range

        if not smooth:
            return filt

        # RTS smoother (offline)
        Ps = torch.stack(Ps); x_preds = torch.stack(x_preds); P_preds = torch.stack(P_preds)
        x_s = xs.clone(); P_s = Ps.clone()
        for t in range(T-2, -1, -1):
            C = Ps[t] @ self.F.T @ torch.linalg.inv(P_preds[t+1])
            x_s[t] = xs[t] + C @ (x_s[t+1] - x_preds[t+1])
            P_s[t] = Ps[t] + C @ (P_s[t+1] - P_preds[t+1]) @ C.T
            # enforce constraints on smoothed too
            x_s[t,1] = torch.clamp(x_s[t,1], -v_clip, v_clip)
            x_s[t,0] = torch.clamp(x_s[t,0], min=0.0)

        return filt, x_s[:, 0]
