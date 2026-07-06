# Data Format Documentation

## PHEME Dataset

### Dataset Overview
PHEME is an English Twitter dataset for rumor detection containing 2,018 source tweets with 34,161 replies across 5 events.

### Required Files Structure

```
pheme_data/
├── content.csv                  # Main data file (imgnum, mid, text, label)
├── post_tweet.csv              # Original tweets (id_str, text, is_rumor, ...)
├── comment_tweet.csv           # Reply tweets (id_str, in_reply_to_status_id_str, text, ...)
└── pheme_images_jpg/           # Image files folder
    ├── 0.jpg
    ├── 1.jpg
    └── ...
```

### File Formats

#### content.csv
Main data table linking images and text with labels.

```csv
imgnum,mid,text,label
0,552784898743099392,"Charlie Hebdo's Last Tweet Before Shootings http://t.co/...",0
1,552786116404072448,"10:28am Charlie Hebdo account mocks ISIS leader...",0
```

**Fields:**
- `imgnum`: Image file identifier (corresponds to {imgnum}.jpg)
- `mid`: Message ID (Twitter status ID)
- `text`: Tweet text content
- `label`: 0 = non-rumor, 1 = rumor

#### post_tweet.csv
Original posts that start propagation chains.

```csv
id_str,text,is_rumor,created_at,user_id,...
552784898743099392,"Original tweet text",0,2015-01-07 10:27:00,123456789,...
```

**Key Fields:**
- `id_str`: Tweet ID (string format to preserve full ID)
- `text`: Tweet content
- `is_rumor`: 0 = false, 1 = true

#### comment_tweet.csv
Reply tweets forming propagation chains.

```csv
id_str,in_reply_to_status_id_str,text,created_at,...
553001234567890123,552784898743099392,"Reply text",2015-01-07 11:30:00,...
```

**Key Fields:**
- `id_str`: Reply tweet ID
- `in_reply_to_status_id_str`: Parent tweet ID (links to post_tweet or another reply)
- `text`: Reply content

### Download

Download PHEME dataset from: [Figshare Link](https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078)

---

## Weibo Dataset

### Dataset Overview
Chinese Weibo dataset containing 1,467 source posts with 528,377 replies.

### Required Files Structure

```
weibo_data/
├── weibo_content.csv           # Main data file (imgnum, mid, text, label)
├── weibo_ori.csv              # Original posts (id_str, text, is_rumor, ...)
├── weibo_reply.csv            # Reply posts (mid, in_reply_to_status_id_str, text, ...)
└── weibo_images_all/          # Image files folder
    ├── 1698.jpg
    ├── 2418.jpg
    └── ...
```

### File Formats

#### weibo_content.csv
Main data table (similar structure to PHEME but in Chinese).

```csv
imgnum,mid,text,label
2418,zw7cXzXxt,"【提醒！花露水属易燃品 涂完别马上做饭】...",1
1698,zd3IOzkpi,"【赶紧牢记下】新交规施行：闯红灯记6分...",1
```

#### weibo_ori.csv
Original Weibo posts.

```csv
id_str,text,is_rumor,created_at,...
zw7cXzXxt,"Original post text in Chinese",1,2019-06-15 10:30:00,...
```

#### weibo_reply.csv
Reply posts forming propagation chains.

```csv
mid,in_reply_to_status_id_str,text,created_at,...
reply_id_001,zw7cXzXxt,"Reply in Chinese",2019-06-15 11:00:00,...
```

### Access

Contact the authors for Weibo dataset access: lukun@hit.edu.cn

---

## Preprocessed Embeddings Format

After running preprocessing scripts, the following JSON files will be generated:

### Text Embeddings (original_text_embeddings.json / weibo_embeddings.json)

```json
{
  "original_posts": {
    "552784898743099392": [0.123, 0.456, 0.789, ...],  // 768-dim vector
    "552786116404072448": [0.234, 0.567, 0.890, ...]
  },
  "reply_posts": {
    "123456789": [0.345, 0.678, 0.901, ...],
    "987654321": [0.456, 0.789, 0.012, ...]
  }
}
```

### Image Embeddings (image_embeddings.json)

```json
{
  "0": [0.111, 0.222, 0.333, ...],     // 768-dim vector (ViT/CLIP)
  "1": [0.444, 0.555, 0.666, ...],
  "2418": [0.777, 0.888, 0.999, ...]
}
```

### Tweet Embeddings for Propagation (tweet_embeddings.json)

Similar structure to text embeddings, used for building propagation trees.

---

## Notes

1. **Image Files**: Images must be named according to the `imgnum` field (e.g., `0.jpg`, `2418.jpg`)
2. **ID Format**: Use string format for tweet IDs to preserve full precision
3. **Label Convention**: 0 = non-rumor/true, 1 = rumor/false
4. **Character Encoding**: UTF-8 for all files (especially important for Weibo Chinese text)
5. **Missing Data**: The preprocessing scripts handle missing images by using zero vectors, but complete data is recommended for best performance

---

## Data Statistics

| Dataset | Language | Sources | Replies | Rumors | Non-Rumors |
|---------|----------|---------|---------|--------|------------|
| PHEME   | English  | 2,018   | 34,161  | 590    | 1,428      |
| Weibo   | Chinese  | 1,467   | 528,377 | 590    | 877        |
