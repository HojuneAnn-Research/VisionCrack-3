import numpy as np
from find_cr import find_cr
def find_rows(g,x,Y,cw,m,span=260):
    sub=g[:, x:x+cw].astype(float)
    cols=np.linspace(0,cw-1,40).astype(int)
    target=m[cols]
    sols=[]
    for top in range(Y-span, Y+1):
        for bot in range(Y, Y+span):
            med=np.median(sub[top:bot+1, cols],axis=0)
            if np.array_equal(med,target):
                if np.array_equal(np.median(sub[top:bot+1],axis=0), m): sols.append((top,bot))
    return sols
if __name__=="__main__":
    import sys,time; t=time.time()
    g,hits,m,cw=find_cr(*sys.argv[1:4]); x,Y=hits[0]
    s=find_rows(g,int(x),int(Y),cw,m); print(s[:10],len(s),time.time()-t)
