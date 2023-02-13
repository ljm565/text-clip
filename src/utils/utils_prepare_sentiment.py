from tqdm import tqdm


           
"""
sentiment type1 utils
"""
def type1_reivew_matcher(lines, file_name):
    all_reviews = []
    for s in tqdm(lines, desc=file_name + ' processing...'):
        assert len(s) == 2
        assert int(s[0]) in [1, 2, 4, 5]

        s[0] = 1 if int(s[0]) > 3 else 0

        all_reviews.append(tuple(s))
    
    return all_reviews



"""
sentiment type2 utils
"""
def type2_reivew_matcher(lines, file_name):
    all_reviews = []
    for s in tqdm(lines, desc=file_name + ' processing...'):
        assert len(s) == 2
        assert int(s[0]) in [0, 1]

        s[0] = int(s[0])

        all_reviews.append(tuple(s))
    
    return all_reviews




"""
sentiment type3 utils
"""
def type3_reivew_matcher(lines, file_name):
    all_reviews = []
    i = 0
    for s in tqdm(lines, desc=file_name + ' processing...'):
        if i == 0:
            i += 1
            continue
        
        assert len(s) == 3
        assert int(s[2]) in [0, 1]

        all_reviews.append((int(s[2]), s[1]))

    return all_reviews