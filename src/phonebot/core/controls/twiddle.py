#!/usr/bin/env python3


class Twiddle():
    """
    Rudimentary implementation of automatic PID tuning.
    """

    def __init__(self, algorithm, params, dp=None):
        self.algorithm_ = algorithm
        self.params_ = params
        self.first_run_ = True
        self.best_err_ = 0.0
        if dp is None:
            #self.dp_ = [.2] * len(params)
            # 0.2 = guess if params not given
            # NOTE : overrides explicit desire to set ki=0, kd=0, etc.
            dp = [0.2*e if e > 0 else 0.2 for e in params]
        self.dp_ = dp
        self.iterations_ = 0

    def run(self):
        if self.first_run_ is True:
            self.best_err_ = self.algorithm_(self.params_)
            self.iterations_ += 1
            self.first_run_ = False
            return self.iterations_, self.params_, self.best_err_

        for i in range(len(self.params_)):
            # Update parameter and run algorithm
            self.params_[i] += self.dp_[i]
            if self.params_[i] < 0:
                self.params_[i] = 0
            err = self.algorithm_(self.params_)

            if err < self.best_err_:
                # Looks good, let's increase the parameters a little bit
                self.best_err_ = err
                self.dp_[i] *= 1.1
            else:
                # The error actually got worse, decrease the parameter
                self.params_[i] -= 2 * self.dp_[i]
                if self.params_[i] < 0:
                    self.params_[i] = 0
                err = self.algorithm_(self.params_)

                if err < self.best_err_:
                    self.best_err_ = err
                    self.dp_[i] *= 1.1
                else:
                    self.params_[i] += self.dp_[i]
                    if self.params_[i] < 0:
                        self.params_[i] = 0
                    self.dp_[i] *= 0.9
        self.iterations_ += 1
        return self.iterations_, self.params_, self.best_err_
