import pandas as pd
import glob
import os, sys

assert len(sys.argv) == 2, "Expect 1 arg: path to _pureklee_ gen dir"
assert os.path.isdir(sys.argv[1])

semuOutDir=os.path.abspath(sys.argv[1])
mktlistfile = os.path.join(semuOutDir, "mutant-0.ktestlist")
df_ktl = []
inittime = os.path.getctime(os.path.join(semuOutDir, "assembly.ll"))
for ktp in glob.glob(os.path.join(semuOutDir, "*.ktest")):
    kttime = os.path.getctime(ktp) - inittime
    assert kttime >= 0, "negative kttime w.r.t assembly.ll, for ktest: "+ ktp
    df_ktl.append({"MutantID":0, 'ktest': os.path.basename(ktp), "ellapsedTime(s)": kttime})

# Handle case where not test is generated
if len(df_ktl) == 0:
    df_ktl = {"MutantID":[], 'ktest':[], "ellapsedTime(s)":[]}

df_ktl = pd.DataFrame(df_ktl)
df_ktl.to_csv(mktlistfile, index=False)
print "#", mktlistfile, "file written."
print "# DONE!"
