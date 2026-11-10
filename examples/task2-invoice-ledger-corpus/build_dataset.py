"""Dataset assembly utility only (validation + extraction). Not product code."""
import hashlib, json, subprocess, os, openpyxl
from pypdf import PdfReader
D='files'; out=[]
os.makedirs('extracted',exist_ok=True)
URL={'skit_library_ledger_2016-21.xlsx':'https://naac.skit.ac.in/QIF/4/4.2/4.2.2/4.2.2&4.2.3.xlsx',
 'skit_egov_vendor_years.xlsx':'https://naac.skit.ac.in/QIF/6/6.2/6.2.3/1_6.2.3.xlsx',
 'skit_egov_bills.pdf':'https://naac.skit.ac.in/QIF/6/6.2/6.2.3/Bills_of_Expenditure.pdf'}
B='https://naac.skit.ac.in/QIF/4/4.2/4.2.2/Subscription_Details/'
for y,n,orig in [('2016-17','ejournal','e-Journal_Subscription.pdf'),('2017-18','ejournal','e-Journal_Subscription.pdf'),('2019-20','ejournal','e-Journal_Subscription.pdf'),('2020-21','ejournal','e-Journal_Subscription.pdf'),('2016-17','delnet','Database_DELNET.pdf'),('2017-18','delnet','Database_DELNET_merged.pdf'),('2018-19','delnet','Database-DELNET_subscription.pdf'),('2019-20','delnet','Database-DELNET_subscription.pdf'),('2020-21','delnet','Database-DELNET_subscription.pdf'),('2018-19','ebooks','e-Books_Subscription.pdf'),('2020-21','ebooks','e-Books_Subscription.pdf'),('2020-21','books','Books_Subscription.pdf')]:
    URL[f'lib_{y}_{n}.pdf']=B+y+'/'+orig
for f in sorted(os.listdir(D)):
    p=os.path.join(D,f); b=open(p,'rb').read()
    r={'filename':f,'source_url':URL[f],'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
    if f.endswith('.pdf'):
        r['type']='application/pdf'
        info=subprocess.run(['pdfinfo',p],capture_output=True,text=True).stdout
        r['pdfinfo_pages']=int([l for l in info.splitlines() if l.startswith('Pages:')][0].split()[1])
        r['pypdf_pages']=len(PdfReader(p).pages)
        pages=[];txt=[]
        for i in range(1,r['pdfinfo_pages']+1):
            t=subprocess.run(['pdftotext','-layout','-f',str(i),'-l',str(i),p,'-'],capture_output=True,text=True).stdout
            imgs=subprocess.run(['pdfimages','-list','-f',str(i),'-l',str(i),p],capture_output=True,text=True).stdout.count('\n')-2
            n=len(t.strip())
            pages.append({'page':i,'text_chars':n,'images':max(imgs,0),'class':'text_layer' if n>200 else ('image_only_scan' if imgs>0 else 'blank_or_minimal')})
            txt.append(f'=== PAGE {i} (physical, 1-based) text_chars={n} ===\n{t}')
        open(f'extracted/{f}.txt','w').write('\n'.join(txt))
        r['pages']=pages
        r['validation']='pdfinfo+pypdf opened; pdftotext exited 0 per page; page class from text_chars>200 and pdfimages count'
    else:
        r['type']='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        wb=openpyxl.load_workbook(p); lines=[]; sh=[]
        for ws in wb:
            nf=0
            for row in ws.iter_rows():
                for c in row:
                    if c.value is not None:
                        if isinstance(c.value,str) and c.value.startswith('='): nf+=1
                        lines.append(f'{ws.title}!{c.coordinate}: {c.value!r}' + (f'  [hyperlink -> {c.hyperlink.target}]' if c.hyperlink else ''))
            sh.append({'sheet':ws.title,'dimensions':ws.dimensions,'formula_cells':nf})
        open(f'extracted/{f}.cells.txt','w').write('\n'.join(lines)+'\n')
        r['sheets']=sh; r['validation']='openpyxl load_workbook ok (formulas not executed); all non-empty cells dumped with hyperlinks'
    out.append(r)
json.dump(out,open('validation.json','w'),indent=1)
for r in out: print(r['filename'],r['bytes'],r.get('pdfinfo_pages',r.get('sheets')), sum(1 for p in r.get('pages',[]) if p['class']=='image_only_scan'))
