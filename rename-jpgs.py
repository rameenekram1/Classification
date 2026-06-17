import os                                                                                                                                        
from pathlib import Path                                                                                                                         
                                                                                                                                                   
base_dir = Path("/home/farmevo/Downloads/test-effi/val")
                                                                                                                                                   
for folder, suffix in [("cn_coty", "0"), ("non_coty", "1")]:                                                                                     
  for file in (base_dir / folder).rglob("*"):                                                                                                  
      if file.is_file():                                                                                                                       
          new_name = file.stem + f"({suffix})" + file.suffix                                                                                   
          file.rename(file.parent / new_name)                                                                                                  
print(f"{folder} done")                                                                                                                      
                                   