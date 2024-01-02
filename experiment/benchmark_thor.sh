for target in "Toaster" "Spatula" "Bread"
do
    CUDA_VISIBLE_DEVICES=$1 xvfb-run --auto-servernum python benchmark_thor.py --target $target --scene FloorPlan1
done

for target in "Painting" "Laptop" "Television"
do
    CUDA_VISIBLE_DEVICES=$1 xvfb-run --auto-servernum python benchmark_thor.py --target $target --scene FloorPlan201
done

for target in "Blinds" "DeskLamp" "Pillow"
do
    CUDA_VISIBLE_DEVICES=$1 xvfb-run --auto-servernum python benchmark_thor.py --target $target --scene FloorPlan301
done

for target in "Mirror" "ToiletPaper" "SoapBar"
do
    CUDA_VISIBLE_DEVICES=$1 xvfb-run --auto-servernum python benchmark_thor.py --target $target --scene FloorPlan401
done

python org_results_thor.py  