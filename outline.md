# Insights from Hotel Reviews

From the project brief: `The main aim of the project is to provide advice for hotel owners on how they can best improve their scores and reviews`

Key narrative points:

* Can we provide hotels with specific advice on which areas they need to improve the most?
* Can we predict which topics a guest will mention in their review based on what we know about them?

# Outline

## 1. Extract, identify and validate topics

Required for next step:

* `csv: topics [topic, topic_id, topic_representations, ...]`
* `csv: review-topic [review_id, topic_ids]`

Dashboard requirements:

* view all topics and their representations in table
* view frequency of topics by hotel, region, ect.

## 2. Extract, identify and validate sentiment of topics

Required for next step:

* `csv: review-topic-sentiment [review_id, topic_id, sentiment -> [-1, 1]]`

Dashboard requirements:

* view overall/hotel-specific topic sentiment
* view distribution (or just range, percentiles, ect.) of sentiment per topic so that we can analyse the spread of the sentiment rather than just the average

## 3. Statistical Analysis

* Can we prove that location/time are NOT independent of which topics are mentioned in a review

Dashboard requirements:

* For statistically significant influences: Input location/time and produce likely topics mentioned

## Conclusion

* Our dashboard enables hotel owners to see which topics are being discussed the most in their reviews and the sentiment assosciated with them. 
* We have shown that the [...] has a significant effect on which topics are mentioned in a review but the [...] does not. This could be used by a hotel to enhance the experience of a guest by delivering a tailored experience.

# Notes

* Or we just merge steps 1 & 2 by using RoBERTa
* Add another step which is generating hotel insights based on our topic-sentiment data using an LLM