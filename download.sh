if [ "$1" = "metaworld" ]; then
    wget -c https://huggingface.co/Po-Chen/flowdiffusion/resolve/main/ckpts/metaworld/model-24.pt
    mkdir -p ckpts/metaworld
    mv model-24.pt ckpts/metaworld/model-24.pt
    echo "Downloaded metaworld model"
elif [ "$1" = "metaworld-DA" ]; then
    wget -c https://huggingface.co/Po-Chen/flowdiffusion/resolve/main/ckpts/metaworld_DA/model-24.pt
    mkdir -p ckpts/metaworld_DA
    mv model-24.pt ckpts/metaworld_DA/model-24.pt
    echo "Downloaded metaworld model"
elif [ "$1" = "ithor" ]; then
    wget -c https://huggingface.co/Po-Chen/flowdiffusion/resolve/main/ckpts/ithor/model-16.pt
    mkdir -p ckpts/ithor
    mv model-16.pt ckpts/ithor/model-16.pt
    echo "Downloaded ithor model"
else 
    echo "Options: {metaworld, metaworld-DA, ithor}"
fi