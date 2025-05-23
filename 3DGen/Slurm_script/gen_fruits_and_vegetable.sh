#!/bin/bash

# List of fruits (without the .txt extension)
fruits=(
    "apples" "apricots" "bananas" "blackberries" "blueberries" "breadfruits" "cantaloupes" "cherries" "clementines"
    "coconuts" "cranberries" "custard_apples" "dates" "dragon_fruits" "durian" "figs" "gooseberries"
    "grapefruits" "grapes" "guavas" "honeydew_melons" "jackfruits" "kiwifruits" "kumquats" "lemons"
    "limes" "lychees" "mangoes" "mandarins" "mangosteens" "mulberries" "nectarines" "oranges" "papayas"
    "passion_fruits" "peaches" "pears" "persimmons" "pineapples" "plums" "pomegranates" "quince" "rambutans"
    "raspberries" "soursop" "star_fruits" "strawberries" "tamarinds" "tangelos" "watermelons"
)

# List of vegetables (without the .txt extension)
vegetables=(
    "acorn_squash" "alfalfa_sprouts" "artichokes" "arugula" "asparagus" "bamboo_shoots" "beets" "bell_peppers"
    "bok_choy" "broccoli" "brussels_sprouts" "butternut_squash" "cabbage" "carrots" "cauliflower" "celery"
    "collard_greens" "corn" "cucumbers" "eggplants" "endive" "fennel" "garlic" "green_beans" "habanero_peppers"
    "jalapeño_peppers" "kale" "leeks" "lettuce" "mushrooms" "mustard_greens" "okra" "onions" "parsnips"
    "peas" "potatoes" "pumpkin" "radishes" "rutabagas" "scallions" "shallots" "spinach" "spaghetti_squash"
    "sweet_potatoes" "swiss_chard" "tomatoes" "turnips" "watercress" "yams" "zucchini"
)

mkdir /lustre/fsn1/projects/rech/tya/ubn15wo/Final_3D_GEN
mkdir /lustre/fsn1/projects/rech/tya/ubn15wo/Final_3D_GEN/Fruits_data/
mkdir /lustre/fsn1/projects/rech/tya/ubn15wo/Final_3D_GEN/Vegetables_data/


# Loop over fruits and submit sbatch commands
for fruit in "${fruits[@]}"; do    
    sbatch generate_images_and_meshes.slurm "/lustre/fswork/projects/rech/tya/ubn15wo/3D_gen/3DMeshes_Generation/Prompt_generation/fruits_final/${fruit}.txt" "/lustre/fsn1/projects/rech/tya/ubn15wo/Final_3D_GEN/Fruits_data/" "${fruit}"
done

# Loop over vegetables and submit sbatch commands
for vegetable in "${vegetables[@]}"; do
    sbatch generate_images_and_meshes.slurm "/lustre/fswork/projects/rech/tya/ubn15wo/3D_gen/3DMeshes_Generation/Prompt_generation/vegetables_final/${vegetable}.txt" "/lustre/fsn1/projects/rech/tya/ubn15wo/Final_3D_GEN/Vegetables_data/" "${vegetable}"
done

