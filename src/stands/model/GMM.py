import scanpy as sc
import numpy as np
from tqdm import tqdm
from scipy.stats import beta ,invgamma, norm


class GMMWithPrior(object):

    def __init__(self, ref_score, random_state=None, max_iter=100, tol=1e-3, prior_beta=[1,10]):

        self.ref_score = ref_score
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.prior_beta = prior_beta
        self.beta = prior_beta
        u = beta.rvs(a=1,b=10,random_state=random_state)
        self.u = [u, 1-u]
        
        self.prior_k = 0.01
        self.k = []
        self.prior_v = 3
        self.v = []
        self.prior_s = np.std(ref_score)
        self.s = []
        self.prior_m = np.mean(ref_score)
        self.m = []
        self.var = [invgamma.rvs(a=self.prior_v/2,scale=self.prior_v*self.prior_s*self.prior_s/2,random_state=random_state)]*2
        self.mean = [norm.rvs(loc=self.prior_m, scale=np.sqrt(self.var[0]), random_state=random_state),
                     norm.rvs(loc=self.prior_m, scale=np.sqrt(self.var[1]), random_state=random_state)]

    def e_step(self, tgt_score):
        p1 = norm.pdf(tgt_score,loc=self.mean[0],scale=np.sqrt(self.var[0]))
        p2 = norm.pdf(tgt_score,loc=self.mean[1],scale=np.sqrt(self.var[1]))

        resp = {}
        resp['Anomalous'] = (self.u[0] * p1)/(self.u[0] * p1 + self.u[1] * p2)
        resp['Normal'] = (self.u[1] * p2)/(self.u[0] * p1 + self.u[1] * p2)
        
        return resp
    
    def m_step(self, tgt_score, resp):
        # update parameters
        z1 = sum(resp['Anomalous'])
        z2 = sum(resp['Normal'])
        self.beta = [self.prior_beta[0] + z1, self.prior_beta[1] + z2]
        self.k = [self.prior_k + z1, self.prior_k + z2]      

        _score1 = sum(resp['Anomalous'] * tgt_score) / z1
        _score2 = sum(resp['Normal'] * tgt_score) / z2
        self.m = [(z1 * _score1 + self.prior_m * self.prior_k) / self.k[0],
                  (z2 * _score2 + self.prior_m * self.prior_k) / self.k[1]]
        self.s = np.sqrt([self.prior_v*(self.prior_s**2)+sum(resp['Anomalous']*(tgt_score**2))+self.prior_k*(self.prior_m**2)-z1*(self.m[0]**2),
                          self.prior_v*(self.prior_s**2)+sum(resp['Normal']*(tgt_score**2))+self.prior_k*(self.prior_m**2)-z2*(self.m[1]**2)])
        self.v = [self.prior_v + z1, self.prior_v + z2]

        self.u = [(self.beta[0]-1)/(sum(self.beta)-2), 1-(self.beta[0]-1)/(sum(self.beta)-2)]
        self.mean = self.m
        self.var = [self.v[0]*(self.s[0]**2)/(self.v[0]+3), self.v[1]*(self.s[1]**2)/(self.v[1]+3)]

    def log_prob(self, tgt_score, resp):
        log_prob1 = sum((np.log(self.u[0])-0.5*np.log(self.var[0])-(tgt_score-self.mean[0])**2/(self.var[0]))*resp['Anomalous'])+\
                    np.log(beta.pdf(x=self.u[0],a=self.prior_beta[0],b=self.prior_beta[1]))+\
                    np.log(norm.pdf(self.mean[0],loc=self.prior_m,scale=np.sqrt(self.var[0]/self.prior_k)))+\
                    np.log(invgamma.pdf(x=self.var[0],a=self.prior_v/2,scale=self.prior_v*(self.prior_s**2)/2))  
        log_prob2 = sum((np.log(self.u[1])-0.5*np.log(self.var[1])-(tgt_score-self.mean[1])**2/(self.var[1]))*resp['Normal'])+\
                    np.log(beta.pdf(x=self.u[1],a=self.prior_beta[1],b=self.prior_beta[0]))+\
                    np.log(norm.pdf(self.mean[1],loc=self.prior_m,scale=np.sqrt(self.var[1]/self.prior_k)))+\
                    np.log(invgamma.pdf(x=self.var[1],a=self.prior_v/2,scale=self.prior_v*(self.prior_s**2)/2))  
        return log_prob1 + log_prob2

    def fit(self, tgt_score):
        tgt_score = np.array(tgt_score)
        tgt_score = np.sort(tgt_score)[::-1]
        resp = self.e_step(tgt_score)
        prob = self.log_prob(tgt_score,resp)

        with tqdm(total=self.max_iter) as t:
            for _ in range(self.max_iter):
                t.set_description(f'Inference Epochs')

                pre_prob = prob
                resp = self.e_step(tgt_score)
                self.m_step(tgt_score, resp)
                prob = self.log_prob(tgt_score, resp)
                if abs(pre_prob-prob) < self.tol:
                    print('GMM-based thresholder has converged.')
                    break    

                t.update(1)
                
        return tgt_score[round(self.u[0]*len(tgt_score))]


if __name__=='__main__':
    ref = norm.rvs(loc=0, scale=0.4,size=1000)
    tgt1 = norm.rvs(loc=0.8, scale=0.3, size=150)
    tgt2 = norm.rvs(loc=0, scale=0.5,size=850)
    tgt = np.concatenate((tgt1, tgt2))
    gmm = GMMWithPrior(ref_score=ref, tol=1e-5,max_iter=1000)
    print(gmm.fit(tgt_score=tgt))

    adata = sc.read('E:/LY/NMBC/slideseq/multiple_anomaly/merge_result.h5ad')
    ref_score = adata[adata.obs['truth']==0].obs['score']
    gmm = GMMWithPrior(ref_score,tol=0.00001)
    a = gmm.fit(tgt_score=adata.obs['score'])  # threshold
    label = [1 if i>=a else 0 for i in adata.obs['score']]
    from sklearn.metrics import f1_score
    f1_score(adata.obs['truth'],label)