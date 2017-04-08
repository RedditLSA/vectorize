# Raw data vectorizer for redditlsa.com
## Instructions
1. Install raw data
```bash
mkdir -p "data"
curl https://storage.googleapis.com/redditlsa/raw_data/sub_pop_all_starting_2015_01.csv > "data/sub_pop_all_starting_2015_01.csv"
curl https://storage.googleapis.com/redditlsa/raw_data/sub_ref_overlap_all_starting_2015_01.csv > "data/sub_ref_overlap_all_starting_2015_01.csv"
```
2. Install requirements
```bash
pip install -r requirements.txt # ideally in a virtualenv
```
3. Run the program
```bash
mkdir -p output
python vectorize.py data output
```
