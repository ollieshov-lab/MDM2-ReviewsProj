# MDM2 Group 16 Project Outline
## Insights from Hotel Reviews

**Brief:** The main aim of the project is to provide advice for hotel owners on how they can best improve their scores and reviews.

**Key questions:**
- Can we provide hotels with specific advice on which areas they need to improve the most?
- Can we predict which topics a guest will mention in their review based on what we know about them?
- Does location or time significantly influence which topics guests mention?

---

## Task Assignments

| Member | Task |
|--------|------|
| John | BERTopic pipeline |
| Tom | NMF comparison |
| Meet | Dynamic topic modelling |
| Nathan | LDA comparison |
| Ollie | Customer profile analysis using statistical methods |

---

## Things We Should Do

- Explore alternatives to BERT, look in topic model wiki page (https://en.wikipedia.org/wiki/Topic_model)
- Find and read similar research (Huw Day's Paper https://www.sciencedirect.com/science/article/pii/S2212094725000982) in order to justify model usage and configurations
- Dynamic topic modelling
- Multi-topic distributions per review

## Things We Could Do

- Dashboard app for exploring data
- Match review dates to historical temperature information
- Predict (sentiment of) topics based on customer data
- Priority and cost adjusted improvement recommendations ← topic sentiment analysis
- Guest profiles to suggest important factors for stay ← if tags are statistically significant influence on topics otherwise individual

---

## Things We Need by the End

### 1. Data & Pre-processing

- Sampling strategy from the 2015–2017 Booking.com dataset
- Pre-processing pipeline (cleaning, language detection, could do multilingual handling)

**Outputs:**
- `csv: processed-reviews [review_id, text, language, city, date, ...]`

---

### 2. Topic Modelling

- BERTopic as primary model with zero-shot topic assignment for hospitality-specific topics
- NMF and LDA to compare with BERTopic
- Dynamic topic modelling to reveal global and periodic trends
- Save BERTopic output to .csv rather than rerunning the pipeline each time

**Outputs:**
- `csv: topics [topic_id, topic_label, top_representations]`
- `csv: review-topic [review_id, topic_id]`

---

### 3. Sentiment Analysis

- Use RoBERTa for sentiment scoring per review and topic pair
- Compare Hugging Face models; find a paper supporting the chosen model

**Outputs:**
- `csv: review-topic-sentiment [review_id, topic_id, sentiment → [-1, 1]]`

---

## Conclusion

- Our pipeline enables hotel owners to see which topics guests discuss most, the sentiment associated with them, and how they compare to competitors in their city.
- We have shown that the [...] has a significant effect on which topics are mentioned in a review but the [...] does not. This could be used by a hotel to enhance the experience of a guest by delivering a tailored experience.

---

## Video Structure

1. **Title Slide**
2. **Introduction**
   - Explanation of our question and what we want to answer in the report/video
   - Real World relevance of the report (how it can be used by hotels to help them improve)
3. **The Dataset(s)**
   - Overview of dataset – size, contents (Hotel name, user nationality, review content, review date, user score, tags) - only include relevant content, if dynamic modelling is used, mention date, if we predict based off tags, mention those, etc
   - If we scrape more data, explain how and why we got those
   - Perhaps explain cleaning of data, using split words
4. **Topic Modelling**
   - Relevance of topic modelling in general
   - How we decided what method to use, comparing BERTopic to either NNMF or Chat-GPT or something else
   - Provide examples of topics they produce
5. **Sentiment Analysis**
   - Why sentiment analysis is used
   - Method used and why (perhaps comparing Vader, RoBERTa or something from Hugging face using accuracies)
   - Explain how sentiment analysis works
6. **Analysis**
7. **Conclusion**
8. **References**

---

## Technical Report

- Focus on reproducibility
- Brief abstract → method → results → short conclusion
- Include link to GitHub repo (https://github.com/ollieshov-lab/MDM2-ReviewsProj) with explanation of what the code does
- Make sure to include: pre-processing steps, model choices and sentiment method.
- Appendix (for material that provides context but is redundant):

---

## Notes

- Merge steps 2 & 3 using RoBERTa directly on topic-filtered review segments
- Add a step using an LLM to generate natural-language hotel topics from the topic-sentiment data
- References: justify BERT with Huw Day's paper and similar literature
