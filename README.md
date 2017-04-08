Vectorize raw data for redditlsa.com
- First, will need to download data
```bash
mkdir -p "data"
curl https://storage.googleapis.com/redditlsa/raw_data/sub_pop_all_starting_2015_01.csv > "data/sub_pop_all_starting_2015_01.csv"
curl https://storage.googleapis.com/redditlsa/vectorized_data/sub_ref_overlap_all_starting_2015_01.csv > "data/sub_ref_overlap_all_starting_2015_01.csv"
```