# Step 0. Change this to your university ID
UID='2001075011'
mkdir -p $UID

# Step 1. (Optional) Any preprocessing step, e.g., downloading pre-trained word embeddings


# Step 2. Train models on on CF-IMDB.
PREF='cfimdb'
python main.py \
    --train "data/${PREF}-train.txt" \
    --dev "data/${PREF}-dev.txt" \
    --test "data/${PREF}-test.txt" \
    --dev_output "${UID}/${PREF}-dev-output.txt" \
    --test_output "${UID}/${PREF}-test-output.txt" \
    --model "${UID}/${PREF}-model.pt"


# Step 3. Prepare submission:
##  3.1. Copy your code to the $UID folder
for file in 'main.py' 'model.py' 'vocab.py' 'setup.py'; do
	cp $file ${UID}/
done
##  3.2. Compress the $UID folder to $UID.zip (containing only .py/.txt/.pdf/.sh files)
##  3.3. Submit the zip file to Canvas! Congrats!