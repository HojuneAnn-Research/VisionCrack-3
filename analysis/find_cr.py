import numpy as np, cv2, pandas as pd, sys, time
def find_cr(img_path, prof_csv, tag):
    g=cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    d=pd.read_csv(prof_csv, encoding="utf-8-sig")
    c=255.0-d[f"{tag}_center_raw"].values; m=255.0-d[f"{tag}_median_raw"].values
    cw=len(c); c=c.astype(np.uint8)
    H,W=g.shape
    # exact row match: search rows/cols where first 40 values match
    key=c[:40]
    hits=[]
    for y in range(H):
        row=g[y]
        # candidate x where row[x]==c[0]
        xs=np.nonzero(row[:W-cw+1]==c[0])[0]
        for x in xs:
            if np.array_equal(row[x:x+40],key) and np.array_equal(row[x:x+cw],c):
                hits.append((x,y))
    return g,hits,m,cw
if __name__=="__main__":
    t=time.time()
    g,hits,m,cw=find_cr(sys.argv[1],sys.argv[2],sys.argv[3]); print(hits[:5],len(hits),cw,time.time()-t)
