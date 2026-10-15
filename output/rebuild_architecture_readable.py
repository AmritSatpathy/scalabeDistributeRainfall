from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import math
import matplotlib
matplotlib.use("Agg")
from matplotlib import mathtext, font_manager, rcParams

ROOT=Path(__file__).parent; OUT=ROOT/"architectures"; LOGO=OUT/"logos"
W,H=1800,1000
REG="/System/Library/Fonts/Supplemental/Times New Roman.ttf"
BOLD="/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"
ITAL="/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf"
INK="#171717"; DARK="#374151"; MID="#5B616B"; RULE="#7C838C"
WHITE="#FFFFFF"; PAPER="#F6F7F8"; PALE="#E9EDF2"; BLUE="#DDE8F2"; GREEN="#E1EEE4"; GOLD="#F2E9DA"
rcParams.update({"mathtext.fontset":"custom","mathtext.rm":"Times New Roman","mathtext.it":"Times New Roman:italic","mathtext.bf":"Times New Roman:bold"})

def ff(n,b=False,i=False): return ImageFont.truetype(ITAL if i else BOLD if b else REG,n)

def base(title,subtitle,h=H):
    im=Image.new("RGB",(W,h),WHITE); d=ImageDraw.Draw(im); d._image=im
    d.line((58,34,W-58,34),fill=INK,width=3)
    d.text((65,55),title,font=ff(48,True),fill=INK)
    d.text((65,112),subtitle,font=ff(25,i=True),fill=MID)
    return im,d

def wrap_lines(d,text,width,font,max_lines=3):
    out=[]
    for para in text.split("\n"):
        words=para.split(); line=""
        for word in words:
            t=(line+" "+word).strip()
            if d.textlength(t,font=font)<=width: line=t
            else:
                if line: out.append(line)
                line=word
        if line: out.append(line)
    return out[:max_lines]

def text_box(d,xy,title,body="",fill=PAPER,outline=DARK,logo=None,ts=31,bs=25):
    x1,y1,x2,y2=xy; d.rounded_rectangle(xy,18,fill=fill,outline=outline,width=3)
    left=x1+18
    if logo:
        li=load_logo(logo,145,105); ly=(y1+y2-li.height)//2
        d._image.paste(li,(x1+18,ly),li); left=x1+li.width+38
    area=max(80,x2-left-18)
    if body:
        tf=ff(ts,True); bf=ff(bs)
        tl=wrap_lines(d,title,area,tf,2); bl=wrap_lines(d,body,area,bf,3)
        th=len(tl)*(ts+4); bh=len(bl)*(bs+4); gap=9; total=th+gap+bh
        yy=y1+(y2-y1-total)/2
        for line in tl:
            d.text((left+(area-d.textlength(line,font=tf))/2,yy),line,font=tf,fill=INK); yy+=ts+4
        yy+=gap
        for line in bl:
            d.text((left+(area-d.textlength(line,font=bf))/2,yy),line,font=bf,fill=MID); yy+=bs+4
    else:
        f=ff(ts,True); ls=wrap_lines(d,title,area,f,3); total=len(ls)*(ts+5); yy=y1+(y2-y1-total)/2
        for line in ls:
            d.text((left+(area-d.textlength(line,font=f))/2,yy),line,font=f,fill=INK); yy+=ts+5

def load_logo(name,mw,mh):
    im=Image.open(LOGO/name).convert("RGBA"); px=im.load()
    for y in range(im.height):
        for x in range(im.width):
            r,g,b,a=px[x,y]
            if name=="docker.png":
                if not (b>r+35 and b>g+18 and a>0): px[x,y]=(r,g,b,0)
            elif min(r,g,b)>235 and max(r,g,b)-min(r,g,b)<18: px[x,y]=(r,g,b,0)
    bb=im.getchannel("A").getbbox()
    if bb: im=im.crop(bb)
    im.thumbnail((mw,mh),Image.Resampling.LANCZOS); return im

def arrow(d,pts,dashed=False,label=None,at=None,width=5):
    if dashed:
        for a,b in zip(pts,pts[1:]):
            dx,dy=b[0]-a[0],b[1]-a[1]; dist=max(1,math.hypot(dx,dy)); ux,uy=dx/dist,dy/dist
            t=0
            while t<dist:
                e=min(dist,t+17); d.line((a[0]+ux*t,a[1]+uy*t,a[0]+ux*e,a[1]+uy*e),fill=DARK,width=width); t+=29
    else: d.line(pts,fill=DARK,width=width,joint="curve")
    a,b=pts[-2],pts[-1]; ang=math.atan2(b[1]-a[1],b[0]-a[0]); ll=19
    d.polygon([b,(b[0]-ll*math.cos(ang-.55),b[1]-ll*math.sin(ang-.55)),(b[0]-ll*math.cos(ang+.55),b[1]-ll*math.sin(ang+.55))],fill=DARK)
    if label:
        x,y=at or pts[len(pts)//2]; f=ff(23,i=True); tw=d.textlength(label,font=f)
        d.text((x-tw/2,y-30),label,font=f,fill=MID)

def boundary(d,xy,label):
    x1,y1,x2,y2=xy
    for x in range(x1,x2,27): d.line((x,y1,min(x+15,x2),y1),fill=RULE,width=3); d.line((x,y2,min(x+15,x2),y2),fill=RULE,width=3)
    for y in range(y1,y2,27): d.line((x1,y,x1,min(y+15,y2)),fill=RULE,width=3); d.line((x2,y,x2,min(y+15,y2)),fill=RULE,width=3)
    d.text((x1+12,y1+9),label,font=ff(23,True),fill=MID)

def footer(d,text,h=H):
    d.line((65,h-61,W-65,h-61),fill="#B7BBC0",width=2); d.text((68,h-48),text,font=ff(21,i=True),fill=MID)

def formula(im,tex,xy,size=30):
    buf=BytesIO(); prop=font_manager.FontProperties(family="Times New Roman",size=size)
    mathtext.math_to_image(tex,buf,prop=prop,dpi=180,format="png",color=INK)
    buf.seek(0); fm=Image.open(buf).convert("RGBA")
    px=fm.load()
    for yy in range(fm.height):
        for xx in range(fm.width):
            r,g,b,a=px[xx,yy]
            if min(r,g,b)>245: px[xx,yy]=(r,g,b,0)
    fm.thumbnail((xy[2]-xy[0],xy[3]-xy[1]),Image.Resampling.LANCZOS)
    im.paste(fm,(int(xy[0]+(xy[2]-xy[0]-fm.width)/2),int(xy[1]+(xy[3]-xy[1]-fm.height)/2)),fm)

def formula_panel(im,d,xy,title,tex,fill=BLUE,size=30,note=""):
    x1,y1,x2,y2=xy
    d.rounded_rectangle(xy,18,fill=fill,outline=DARK,width=3)
    f=ff(30,True); tw=d.textlength(title,font=f); d.text((x1+(x2-x1-tw)/2,y1+24),title,font=f,fill=INK)
    formula(im,tex,(x1+20,y1+78,x2-20,y2-(62 if note else 26)),size)
    if note:
        nf=ff(25,i=True); nw=d.textlength(note,font=nf); d.text((x1+(x2-x1-nw)/2,y2-52),note,font=nf,fill=MID)

def save(im,name): im.save(OUT/name,dpi=(240,240),optimize=True)

# A1 uses a taller canvas so the report can display a readable vertical research flow.
im,d=base("A1. End-to-end research architecture","A layered view of data engineering, analytical tasks, model training and systems evidence",1180)
boundary(d,(65,175,1735,1050),"Singapore rainfall research system · 2017–2024")
d.text((90,222),"DATA FOUNDATION",font=ff(25,True),fill=MID)
for x,t,b,l in [(100,"Annual observations","station-time CSV files","apache-parquet.png"),(610,"Spark quality layer","schema · bounds · deduplication","apache-spark.png"),(1120,"Reusable feature store","year-partitioned Parquet","apache-parquet.png")]:
    text_box(d,(x,260,x+410,425),t,b,BLUE if x==610 else PAPER,logo=l)
arrow(d,[(510,343),(610,343)]); arrow(d,[(1020,343),(1120,343)])
d.text((90,483),"ANALYTICAL BRANCHES",font=ff(25,True),fill=MID)
text_box(d,(115,525,465,685),"Task 1","seasonal · diurnal · spatial",fill=GOLD)
text_box(d,(575,525,925,685),"Task 1 evidence","figures · tables · caveats",fill=PAPER)
text_box(d,(115,755,465,915),"Task 2","temporal windows · thresholds",fill=GREEN)
text_box(d,(575,735,925,935),"Predictive models","Spark MLlib + bounded PyTorch",fill=PAPER,logo="apache-spark.png",ts=30,bs=23)
text_box(d,(1110,560,1645,800),"Academic synthesis","weather findings · predictive evaluation · deployment limits",fill=PALE)
arrow(d,[(1325,425),(1325,480),(290,480),(290,525)])
arrow(d,[(1325,425),(1325,470),(1050,470),(1050,720),(290,720),(290,755)])
arrow(d,[(465,605),(575,605)]); arrow(d,[(465,835),(575,835)])
arrow(d,[(925,605),(1015,605),(1015,650),(1110,650)])
arrow(d,[(925,835),(1015,835),(1015,715),(1110,715)])
text_box(d,(1110,850,1645,980),"Matched systems study","Local · Standalone · Docker",fill=BLUE)
arrow(d,[(1378,850),(1378,785)],dashed=True,label="runtime evidence",at=(1490,830))
footer(d,"Temporal contract: 2017–2023 training · 2024 test · small summaries only are collected",1180)
save(im,"A1_end_to_end_workflow.png")

im,d=base("A2. Distributed data-engineering pipeline","Partition-aware transformations and bounded collection")
steps=[("Parse","types + timestamps"),("Validate","bounds + deduplicate"),("Partition","clean Parquet by year"),("Window","station history"),("Expand","rainfall threshold q")]
for i,(t,b) in enumerate(steps):
    x=65+i*345; text_box(d,(x,275,x+285,475),t,b,BLUE if i in (2,3) else PAPER,logo="apache-spark.png" if i==3 else None,ts=31,bs=24)
    if i<4: arrow(d,[(x+285,375),(x+345,375)],label="shuffle" if i in (1,2) else None,at=(x+315,350))
text_box(d,(320,650,760,835),"Compact summaries","Task 1 plots and tables",PALE)
text_box(d,(1040,650,1480,835),"Distributed matrices","Task 2 train and test frames",PALE)
arrow(d,[(550,475),(550,650)]); arrow(d,[(1500,475),(1500,565),(1260,565),(1260,650)])
footer(d,"Materialisation separates expensive temporal feature construction from repeated model fits")
save(im,"A2_data_engineering_pipeline.png")

im,d=base("A3. Local Spark execution topology","Multithreaded execution within one JVM and one physical host")
boundary(d,(90,190,1710,885),"Single laptop · one operating-system process")
text_box(d,(170,345,520,630),"Spark driver","DAG · scheduler · context",BLUE,logo="apache-spark.png")
text_box(d,(720,265,1110,485),"Executor thread pool","local[2 / 4 / 8 / *]",PAPER)
text_box(d,(720,605,1110,825),"Shared JVM heap","cache · serialization · GC",PAPER)
text_box(d,(1330,265,1630,485),"CPU cores","same host",PALE)
text_box(d,(1330,605,1630,825),"Local disks","input · shuffle · output",PALE)
arrow(d,[(520,440),(720,375)],dashed=True,label="schedule",at=(620,370)); arrow(d,[(1110,375),(1330,375)],label="tasks",at=(1220,350))
arrow(d,[(915,485),(915,605)],dashed=True,label="memory",at=(1010,565)); arrow(d,[(1110,715),(1210,715),(1210,715),(1330,715)],label="I/O",at=(1215,690))
footer(d,"Interpretation: local mode provides parallel threads, not networked worker nodes")
save(im,"A3_local_spark_topology.png")

im,d=base("A4. Spark Standalone execution topology","Driver, master and executor processes sharing one physical laptop")
boundary(d,(65,185,1735,900),"Single physical host")
text_box(d,(125,390,440,655),"Driver","requests executors · sends tasks",BLUE,logo="apache-spark.png")
text_box(d,(680,235,1045,440),"Standalone master","worker registry · resource offers",PALE,logo="apache-spark.png")
text_box(d,(1280,225,1665,445),"Worker 1","executor · cores · heap",PAPER,logo="apache-spark.png")
text_box(d,(1280,640,1665,860),"Workers 2–4","executors · cores · heaps",PAPER,logo="apache-spark.png")
arrow(d,[(440,455),(555,455),(555,330),(680,330)],dashed=True,label="resource request",at=(555,305))
arrow(d,[(1045,335),(1160,335),(1160,335),(1280,335)],dashed=True,label="allocate",at=(1160,310))
arrow(d,[(1045,380),(1130,380),(1130,750),(1280,750)],dashed=True,label="register",at=(1200,725))
arrow(d,[(440,540),(570,540),(570,525),(1195,525),(1195,405),(1280,405)],label="task dispatch",at=(850,500))
arrow(d,[(440,590),(540,590),(540,875),(1710,875),(1710,750),(1665,750)],label="task dispatch",at=(900,850))
arrow(d,[(1665,445),(1710,445),(1710,640),(1665,640)],label="shuffle",at=(1740,555))
footer(d,"Tested layouts: 2 × 2 cores and 4 × 2 cores; process distribution remains single-host")
save(im,"A4_standalone_topology.png")

im,d=base("A5. Dockerised Spark execution topology","Prominent transparent Docker marks, bridge networking and a shared bind mount")
boundary(d,(65,185,1735,900),"Docker engine on one physical laptop")
text_box(d,(115,260,555,520),"Master container","Spark master + submitted driver",BLUE,logo="docker.png")
text_box(d,(720,235,1160,500),"Worker container 1","Spark worker + executor",PAPER,logo="docker.png")
text_box(d,(1285,235,1685,500),"Workers 2–3","Spark worker + executor",PAPER,logo="docker.png")
text_box(d,(260,680,1540,845),"Shared bind-mounted project volume","data · Parquet · models · benchmark logs",PALE,logo="apache-parquet.png")
arrow(d,[(555,365),(720,365)],dashed=True,label="bridge RPC",at=(638,340)); arrow(d,[(1160,365),(1285,365)],label="shuffle",at=(1222,340))
arrow(d,[(335,520),(335,680)],label="read / write",at=(430,625)); arrow(d,[(940,500),(940,590),(900,590),(900,680)],label="shared path",at=(1020,620))
arrow(d,[(1485,500),(1485,680)],label="shared path",at=(1570,625))
footer(d,"Container isolation improves deployment repeatability; it does not create multi-host scale-out")
save(im,"A5_docker_topology.png")

im,d=base("A6. Common Spark MLlib pipeline","Readable estimator flow with optional scaling and an explicit fit path")
items=[("Station ID","StringIndexer"),("Features","VectorAssembler"),("Scaling","MinMaxScaler"),("Estimator","fit: 2017–2023"),("Scoring","transform: 2024"),("Evaluation","ROC-AUC · PR-AUC")]
for i,(t,b) in enumerate(items):
    x=45+i*295; text_box(d,(x,310,x+245,520),t,b,BLUE if i==3 else PAPER,logo="apache-spark.png" if i==3 else None,ts=29,bs=23)
    if i<5: arrow(d,[(x+245,415),(x+295,415)])
arrow(d,[(635,310),(635,225),(930,225),(930,310)],dashed=True,label="tree models bypass scaling",at=(780,205))
text_box(d,(760,690,1080,845),"Training label","next-30-minute event",PALE)
arrow(d,[(920,690),(920,520)],dashed=True,label="fit",at=(965,625))
text_box(d,(1260,680,1690,855),"Output contract","probability where available; SVC returns a margin",PALE)
arrow(d,[(1640,520),(1640,680)])
footer(d,"MLlib is represented by the Apache Spark mark because it has no separate official logo")
save(im,"A6_common_mllib_pipeline.png")

im,d=base("A7. Controlled experiment design","A matched protocol for credible deployment and scaling comparisons")
text_box(d,(85,330,470,610),"Frozen contract","row IDs · features · split · model · seed",BLUE)
for y,t,b,l in [(220,"Local mode","one JVM","apache-spark.png"),(455,"Standalone","separate executors","apache-spark.png"),(690,"Docker","container workers","docker.png")]:
    text_box(d,(650,y,1025,y+180),t,b,PAPER,logo=l); arrow(d,[(470,470),(560,470),(560,y+90),(650,y+90)],dashed=True)
text_box(d,(1215,330,1585,610),"Measurement","warm-up · repeated runs · median + range",PALE)
text_box(d,(1160,710,1690,865),"Joint report","runtime · efficiency · shuffle · spill · GC · model quality",GREEN)
for y in (310,545,780): arrow(d,[(1025,y),(1120,y),(1120,470),(1215,470)])
arrow(d,[(1400,610),(1400,710)])
footer(d,"The same data, resources, partitions and evaluation set are required for causal comparison")
save(im,"A7_controlled_experiment_design.png")

# Model atlas: fewer words, larger labels and proper mathematical notation.
im,d=base("M1. Gradient-Boosted Trees","Sequential residual correction in an additive tree ensemble")
text_box(d,(70,385,330,625),"Feature vector","unscaled",PAPER)
for i,x in enumerate((520,770,1020,1270)):
    text_box(d,(x,310,x+175,545),f"Tree {i+1}","depth ≤ 5",BLUE if i==0 else PALE,ts=29,bs=24)
    if i<3: arrow(d,[(x+175,425),(x+250,425)],dashed=True,label="residual",at=(x+212,400))
arrow(d,[(330,505),(430,505),(430,425),(520,425)])
text_box(d,(1510,385,1730,625),"Probability","sigmoid of summed margin",GREEN,ts=28,bs=23)
for x in (608,858,1108,1358): arrow(d,[(x,545),(x,710),(1460,710),(1460,505),(1510,505)])
d.text((610,800),"40 rounds · learning rate 0.1 · subsampling 0.70",font=ff(27,i=True),fill=MID)
footer(d,"Primary non-linear Spark MLlib model")
save(im,"M1_gbt_architecture.png")

im,d=base("M2. Random Forest","Parallel bagging, feature subsampling and probability averaging")
text_box(d,(70,385,330,625),"Feature vector","unscaled",PAPER)
for i,(x,y) in enumerate(((510,240),(820,240),(510,650),(820,650))):
    text_box(d,(x,y,x+235,y+190),f"Tree sample {i+1}","bootstrap + feature subset",PALE,ts=27,bs=22)
    arrow(d,[(330,505),(420,505),(420,y+95),(x,y+95)],dashed=True)
text_box(d,(1190,385,1455,625),"Average votes","60 depth-8 trees",BLUE)
for x,y in ((745,335),(1055,335),(745,745),(1055,745)): arrow(d,[(x,y),(1120,y),(1120,505),(1190,505)])
text_box(d,(1570,385,1735,625),"Output","class probability",GREEN,ts=27,bs=22); arrow(d,[(1455,505),(1570,505)])
footer(d,"Bagged-tree comparison model")
save(im,"M2_random_forest_architecture.png")

im,d=base("M3. Logistic Regression","Regularised linear score with a calibrated link function")
text_box(d,(80,385,360,625),"Scaled features","input vector",PAPER)
formula_panel(im,d,(590,315,1050,695),"Linear score",r"$z=\beta_0+\sum_{j=1}^{p}\beta_j x_j$",BLUE,34,"L2 regularisation")
formula_panel(im,d,(1220,315,1595,695),"Sigmoid link",r"$P(y=1\mid x)=\frac{1}{1+e^{-z}}$",PALE,29)
text_box(d,(1650,385,1755,625),"Output","p + class",GREEN,ts=26,bs=22)
arrow(d,[(360,505),(590,505)]); arrow(d,[(1050,505),(1220,505)]); arrow(d,[(1595,505),(1650,505)])
footer(d,"Subscripts and superscripts are typeset mathematically to avoid missing-glyph boxes")
save(im,"M3_logistic_regression_architecture.png")

im,d=base("M4. Linear Support Vector Classifier","Maximum-margin separation with an explicit decision score")
text_box(d,(80,385,360,625),"Scaled features","input vector",PAPER)
text_box(d,(590,315,1060,695),"Maximum-margin fit","hinge loss + regularisation",BLUE)
arrow(d,[(360,505),(590,505)]); arrow(d,[(1060,505),(1160,505)])
for off in (-55,0,55): d.line((1190+off,720,1515+off,315),fill=INK if off==0 else RULE,width=7 if off==0 else 3)
for x,y,c in [(1215,430,"#6B8EAD"),(1310,575,"#6B8EAD"),(1400,655,"#6B8EAD"),(1400,390,"#B07745"),(1500,500,"#B07745"),(1320,300,"#B07745")]: d.ellipse((x-16,y-16,x+16,y+16),fill=c,outline=INK,width=2)
text_box(d,(1580,385,1750,625),"Output","margin + class",PALE,ts=27,bs=22); arrow(d,[(1545,505),(1580,505)])
footer(d,"Linear SVC does not provide a native probability without a separate calibration step")
save(im,"M4_linear_svc_architecture.png")

im,d=base("M5. Spark MLlib multilayer perceptron","Feed-forward architecture: input → 32 → 16 → 2")
layers=[(230,7,"Input"),(700,6,"Hidden 32"),(1160,5,"Hidden 16"),(1580,2,"Output 2")]; coords=[]
for x,n,label in layers:
    ys=[285+i*(430/(n-1 if n>1 else 1)) for i in range(n)]; coords.append([(x,y) for y in ys])
    for y in ys: d.ellipse((x-22,y-22,x+22,y+22),fill=BLUE if "Hidden" in label else PALE,outline=INK,width=3)
    tw=d.textlength(label,font=ff(30,True)); d.text((x-tw/2,790),label,font=ff(30,True),fill=INK)
for a,b in zip(coords,coords[1:]):
    for p in a:
        for q in b: d.line((p,q),fill="#B9BEC5",width=1)
d.text((560,860),"Min–max scaling · nonlinear hidden activations · 100 iterations",font=ff(27,i=True),fill=MID)
footer(d,"Spark MLlib neural baseline")
save(im,"M5_mlp_architecture.png")

im,d=base("M6. Factorization Machines","Linear effects combined with learned low-rank interactions")
text_box(d,(70,385,340,625),"Scaled features","input vector",PAPER)
formula_panel(im,d,(570,235,1010,460),"Linear contribution",r"$w_0+\sum_{i=1}^{p}w_i x_i$",PALE,30)
formula_panel(im,d,(570,625,1010,850),"Pairwise interactions",r"$\sum_{i<j}\langle v_i,v_j\rangle x_i x_j$",BLUE,27,"rank 8 latent factors")
text_box(d,(1240,385,1500,625),"Combined logit","linear + interactions",PALE)
text_box(d,(1615,385,1750,625),"Output","sigmoid p",GREEN,ts=27,bs=22)
arrow(d,[(340,505),(460,505),(460,347),(570,347)]); arrow(d,[(340,505),(460,505),(460,737),(570,737)])
arrow(d,[(1010,347),(1120,347),(1120,480),(1240,480)]); arrow(d,[(1010,737),(1120,737),(1120,545),(1240,545)]); arrow(d,[(1500,505),(1615,505)])
footer(d,"Interaction-aware model with properly typeset indices")
save(im,"M6_factorization_machine_architecture.png")

im,d=base("M7. LSTM sequence model","Recent rainfall dynamics fused with static context on one device")
for i,x in enumerate((70,230,390,550)):
    text_box(d,(x,380,x+120,590),f"t − {3-i}","rain",PALE,ts=28,bs=23)
    if i<3: arrow(d,[(x+120,485),(x+160,485)])
text_box(d,(820,315,1190,655),"LSTM encoder","1 layer · hidden size 32",BLUE,logo="pytorch.png")
text_box(d,(820,735,1190,875),"Static context","station · time · threshold",PAPER)
text_box(d,(1360,385,1595,625),"Fusion head","dense 32 · ReLU · dropout",PALE,ts=28,bs=23)
text_box(d,(1665,385,1760,625),"Output","sigmoid p",GREEN,ts=25,bs=21)
arrow(d,[(670,485),(820,485)]); arrow(d,[(1190,485),(1360,485)]); arrow(d,[(1190,805),(1280,805),(1280,555),(1360,555)]); arrow(d,[(1595,505),(1665,505)])
footer(d,"Supplementary single-device PyTorch experiment; not distributed Spark training")
save(im,"M7_lstm_architecture.png")

im,d=base("M8. Temporal convolutional network","Dilated temporal filters, global pooling and context fusion")
text_box(d,(60,385,320,625),"Rain sequence","batch × time × 1",PAPER)
text_box(d,(500,245,820,465),"Temporal Conv 1","32 channels · kernel 3",BLUE,logo="pytorch.png")
text_box(d,(500,665,820,855),"Static context","station · time · threshold",PAPER)
text_box(d,(1000,245,1300,465),"Dilated Conv 2","32 channels · dilation 2",BLUE)
text_box(d,(1000,665,1300,855),"Global pooling","fixed-length state",PALE)
text_box(d,(1430,385,1625,625),"Fusion head","context + dense",PALE,ts=28,bs=22)
text_box(d,(1690,385,1770,625),"Output","p",GREEN,ts=24,bs=21)
arrow(d,[(320,505),(410,505),(410,355),(500,355)]); arrow(d,[(820,355),(1000,355)]); arrow(d,[(1150,465),(1150,665)])
arrow(d,[(1300,760),(1370,760),(1370,555),(1430,555)]); arrow(d,[(820,760),(900,760),(900,555),(1430,555)]); arrow(d,[(1625,505),(1690,505)])
footer(d,"Supplementary single-device PyTorch experiment")
save(im,"M8_tcn_architecture.png")

print("Rebuilt 15 architecture figures for report-scale readability")
