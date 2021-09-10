import os

imgs = os.listdir("./imagenet")
print(imgs)

with open("./autogen.mjs", "w") as f:
    f.write("""export const imageNames ="""+str(imgs))