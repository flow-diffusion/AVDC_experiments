if [ "$1" = "metaworld" ]; then
    wget -c https://huggingface.co/Po-Chen/flowdiffusion/resolve/main/ckpts/metaworld/model-24.pt
    mkdir -p ckpts/mw
    mv model-24.pt ckpts/mw/model-24.pt
    echo "Downloaded metaworld model"
elif [ "$1" = "ithor" ]; then
    wget -c https://huggingface.co/Po-Chen/flowdiffusion/resolve/main/ckpts/ithor/model-30.pt
    mkdir -p ckpts/thor
    mv model-30.pt ckpts/thor/model-30.pt
    echo "Downloaded ithor model"
elif [ "$1" = "diffusion-policy" ]; then
    wget -c https://huggingface.co/Po-Chen/flowdiffusion/resolve/main/ckpts/diffusion_policy/model-10.pt
    mkdir -p ckpts/diffusion_policy
    mv model-10.pt ckpts/diffusion_policy/model-10.pt
    echo "Downloaded diffusion-policy model"
else 
    echo "Options: {metaworld, ithor, diffusion-policy}"
fi