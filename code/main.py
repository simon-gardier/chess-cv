import time
from program import *

program = Program()
program.run()
program.close()
t1 = time.time()
print(f"⏱️  Total time : {t1-t0:.2f}s")