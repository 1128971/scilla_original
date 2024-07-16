from flask import Flask, render_template, request, redirect, url_for
import time
import numpy as np
import requests
import re
import praw
import os
import seaborn as sns
import io
from praw.models import MoreComments
from fake_useragent import UserAgent
from time import sleep
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from selenium.webdriver.common.by import By
from sklearn.cluster import KMeans
from selenium.common.exceptions import JavascriptException
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import csv
import mplcyberpunk
from urllib.parse import urljoin
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
import vk_api
import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.keys import Keys
from telebot import types
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram import ReplyKeyboardMarkup
from googleapiclient.discovery import build
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import pipeline
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from geopy.geocoders import Nominatim
import folium
from folium.plugins import HeatMap

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)

model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def load_reddit_credentials():
    with open('credentials.json', 'r') as f:
        credentials = json.load(f)
    return credentials['REDDIT']

def get_comments(video_url):
    video_id = video_url.split("v=")[1].split("&")[0]
    api_key = "AIzaSyDlzgevoqWt1gZowr2hJAd474gTffo0ILM"
    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []
    likes = []
    next_page_token = None
    while True:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token
        ).execute()
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay'].replace('\n', ' ')
            like_count = item['snippet']['topLevelComment']['snippet']['likeCount']
            comments.append(comment)
            likes.append(like_count)
        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break
    return comments, likes

def get_habr_comments(url):
    if '/comments' not in url:
        if url[-1] == '/':
            url += 'comments'
        else:
            url += '/comments'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    sections = soup.find('div', {'class': 'tm-comments__tree'})
    if not sections:
        return []
    rows = []
    for section in sections.find_all('article', {'class': 'tm-comment-thread__comment'}):
        if not section.find('a', {'class': 'tm-user-info__userpic'}):
            continue
        user_url = 'https://habr.com' + section.find('a', {'class': 'tm-user-info__userpic'}).get('href')
        user = section.find('a', {'class': 'tm-user-info__username'}).text.strip()
        text = section.find('div', {'class': 'tm-comment__body-content'}).text.strip().replace('\n', ' ')
        likes = int(section.find('span', {'class': 'tm-votes-meter__value'}).text.strip())
        rows.append({
            'comments': text,
            'user': user,
            'likes': likes,
            'user_url': user_url,
            'source': 'Habr'
        })
    return rows

def get_reddit_comments(url, settings):
    comments = []
    reddit = praw.Reddit(
        client_id=settings['client_id'],
        client_secret=settings['client_secret'],
        user_agent=settings['user_agent'],
        username=settings['username'],
        password=settings['password']
    )
    post = reddit.submission(url=url)
    post.comments.replace_more(limit=None)
    for comment in post.comments.list():
        if comment.author is not None:
            comments.append({
                'comments': comment.body,
                'user': comment.author.name,
                'likes': comment.score,
                'user_url': f'https://www.reddit.com/user/{comment.author.name}',
                'source': 'Reddit'
            })
    return comments

def parse_steam_comments(url):
    pattern = r'app/(\d+)/'
    game_id = re.search(pattern, url).group(1)
    template_with_language = 'https://steamcommunity.com/app/{}/reviews/?browsefilter=toprated&filterLanguage=russian'
    url = template_with_language.format(game_id)
    driver = webdriver.Chrome()
    driver.get(url)
    reviews = []
    last_position = driver.execute_script("return window.pageYOffset;")
    running = True
    while running:
        cards = driver.find_elements(By.CLASS_NAME, 'apphub_Card')
        for card in cards[-20:]:
            profile_url = card.find_element(By.XPATH, './/div[@class="apphub_friend_block"]/div/a[2]').get_attribute('href')
            user_name = card.find_element(By.XPATH, './/div[@class="apphub_friend_block"]/div/a[2]').text
            review_content = card.find_element(By.XPATH, './/div[@class="apphub_CardTextContent"]').text.strip().replace('\n', ' ')
            thumb_text = card.find_element(By.XPATH, './/div[@class="reviewInfo"]/div[2]').text
            reviews.append({
                "comments": review_content,
                'user': user_name,
                'likes': thumb_text,
                'user_url': profile_url,
                'source': 'Steam'
            })
        scroll_attempt = 0
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            sleep(0.5)
            curr_position = driver.execute_script("return window.pageYOffset;")
            if curr_position == last_position:
                scroll_attempt += 1
                sleep(0.5)
                if scroll_attempt >= 3:
                    running = False
                    break
            else:
                last_position = curr_position
                break
    driver.close()
    return reviews

def get_steam_comments(urls):
    df = pd.DataFrame(columns=['comments', 'user', 'likes', 'user_url', 'source'])
    for url in urls:
        comments = parse_steam_comments(url)
        if comments:
            df = pd.concat([pd.DataFrame(comments), df])
    df.reset_index(drop=True, inplace=True)
    return df

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('russian'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return ' '.join(lemmatized_tokens)

def create_wordcloud_from_comments():
    with open("comments.csv", "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        comments = [row[0].replace('\n', ' ') for row in reader]
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(comments))
    return wordcloud

def analyze_comments():
    df = pd.read_csv("comments.csv")
    df['comments'] = df['comments'].str.replace('\n', ' ')
    result_df = pd.DataFrame(columns=['Комментарий', 'Оценка', 'Количество лайков', 'Эмоциональный окрас'])
    results = []
    for index, row in df.iterrows():
        comment = row['comments']
        likes = row['likes']
        result = model(comment[:512])  # Обрезаем комментарий до 512 токенов
        sentiment = result[0]['label']
        score = result[0]['score']
        results.append({'Комментарий': comment, 'Оценка': sentiment, 'Количество лайков': likes, 'Эмоциональный окрас': score})
    result_df = pd.DataFrame(results)
    return result_df

# Загрузка списка городов РФ
cities = pd.read_csv('russian_cities.csv')  # Файл должен содержать список городов и их регионы

def find_cities_in_text(text, cities_list):
    found_cities = []
    for city in cities_list:
        if city.lower() in text.lower():
            found_cities.append(city)
    return found_cities

def create_heatmap():
    # Чтение комментариев
    comments_df = pd.read_csv("comments.csv")
    cities_df = pd.read_csv("russian_cities.csv")

    # Инициализация геолокатора
    geolocator = Nominatim(user_agent="geoapiExercises")

    # Обработка каждого комментария
    city_mentions = []
    for comment in comments_df['comments']:
        mentioned_cities = find_cities_in_text(comment, cities_df['city'])
        city_mentions.extend(mentioned_cities)

    # Геокодирование упомянутых городов
    city_coordinates = []
    for city in city_mentions:
        location = geolocator.geocode(city + ', Russia')
        if location:
            city_coordinates.append([location.latitude, location.longitude])

    # Создание тепловой карты
    heat_map = folium.Map(location=[55.751244, 37.618423], zoom_start=4)
    if city_coordinates:
        HeatMap(city_coordinates).add_to(heat_map)
    heatmap_path = os.path.join('static', 'heatmap.html')
    heat_map.save(heatmap_path)
    return heatmap_path

def create_clusters():
    with open("comments.csv", "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        comments = [row[0].replace('\n', ' ') for row in reader]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(comments)
    num_clusters = 5
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
    labels = kmeans.labels_
    comments_df = pd.DataFrame({'Комментарий': comments, 'Кластер': labels})
    comments_df['Комментарий'] = comments_df['Комментарий'].str.replace('\n', ' ').str.strip()
    comments_df.to_csv('clusters.csv', index=False, encoding='utf-8')
    return comments_df

def max_likes():
    df = pd.read_csv("comments.csv")
    df['comments'] = df['comments'].str.replace('\n', ' ')
    top_liked_comments = df.nlargest(50, 'likes')
    top_liked_comments.to_csv('top_liked_comments.csv', index=False, encoding='utf-8')
    return top_liked_comments

def generate_histogram():
    df = pd.read_csv("comments.csv")
    df['comments'] = df['comments'].apply(preprocess_text)
    all_words = [word for tokens in df['comments'] for word in tokens.split()]
    word_freq = Counter(all_words).most_common(10)
    words, freqs = zip(*word_freq)  # Распаковываем слова и их частоты
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(words, freqs, color='skyblue', edgecolor='black')
    ax.set_xlabel('Слова')
    ax.set_ylabel('Частота')
    ax.set_title('Топ-10 частоиспользуемых слов в комментариях')
    plt.xticks(rotation=45)
    plt.grid(True)
    histogram_path = os.path.join('static', 'histogram.png')
    plt.savefig(histogram_path)
    plt.close(fig)  # Закрытие фигуры после сохранения
    return histogram_path

def search_word_in_comments(word):
    df = pd.read_csv("comments.csv")
    df['comments'] = df['comments'].str.replace('\n', ' ')
    word = word.lower()
    matched_comments = df[df['comments'].str.contains(word, case=False, na=False)]
    matched_comments.to_csv('matched_comments.csv', index=False, encoding='utf-8')
    return matched_comments

def sentiment_distribution():
    df = pd.read_csv("comments.csv")
    if 'sentiment' not in df.columns:
        df['sentiment'] = df['comments'].apply(lambda x: model(x[:512])[0]['label'])  # Обрезаем комментарий до 512 токенов
    sentiments = df['sentiment']
    sentiment_counts = pd.Series(sentiments).value_counts()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
    plt.xlabel('Тональность')
    plt.ylabel('Количество комментариев')
    plt.title('Распределение тональности комментариев')
    plt.grid(True)
    
    distribution_path = os.path.join('static', 'sentiment_distribution.png')
    plt.savefig(distribution_path)
    plt.close()
    return distribution_path

def comment_length_distribution():
    df = pd.read_csv("comments.csv")
    if 'sentiment' not in df.columns:
        df['sentiment'] = df['comments'].apply(lambda x: model(x[:512])[0]['label'])  # Обрезаем комментарий до 512 токенов
    df['length'] = df['comments'].apply(len)

    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='length', hue='sentiment', multiple='stack', palette='viridis')
    plt.xlabel('Длина комментария')
    plt.ylabel('Количество комментариев')
    plt.title('Распределение длины комментариев по тональности')
    plt.grid(True)

    distribution_path = os.path.join('static', 'comment_length_distribution.png')
    plt.savefig(distribution_path)
    plt.close()
    return distribution_path

def comment_length_stats():
    df = pd.read_csv("comments.csv")
    if 'sentiment' not in df.columns:
        df['sentiment'] = df['comments'].apply(lambda x: model(x[:512])[0]['label'])  # Обрезаем комментарий до 512 токенов
    df['length'] = df['comments'].apply(len)
    stats = df.groupby('sentiment')['length'].describe()
    return stats

def bribe_comment_count():
    df = pd.read_csv("comments.csv")
    df['comments'] = df['comments'].str.lower()
    bribe_keywords = ['взятка', 'коррупция', 'подкуп']
    
    df['contains_bribe'] = df['comments'].apply(lambda x: any(word in x for word in bribe_keywords))
    bribe_counts = df['contains_bribe'].value_counts()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=bribe_counts.index, y=bribe_counts.values, palette='viridis', hue=bribe_counts.index)
    plt.xlabel('Содержит упоминание взятки')
    plt.ylabel('Количество комментариев')
    plt.title('Количество комментариев по взяткам')
    plt.grid(True)
    
    bribe_count_path = os.path.join('static', 'bribe_comment_count.png')
    plt.savefig(bribe_count_path)
    plt.close()
    return bribe_count_path

def corruption_clusters():
    df = pd.read_csv("comments.csv")
    df['comments'] = df['comments'].str.lower()
    corruption_keywords = ['коррупция', 'взятка', 'подкуп', 'откуп']
    
    df['contains_corruption'] = df['comments'].apply(lambda x: any(word in x for word in corruption_keywords))
    df['cluster'] = KMeans(n_clusters=5, random_state=0).fit_predict(CountVectorizer().fit_transform(df['comments']))

    corruption_counts = df[df['contains_corruption']].groupby('cluster').size()

    plt.figure(figsize=(10, 6))
    if len(corruption_counts) > 0:
        corruption_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis', len(corruption_counts)))
    plt.title('Кластеры обсуждений о коррупции')
    plt.ylabel('')
    
    corruption_clusters_path = os.path.join('static', 'corruption_clusters.png')
    plt.savefig(corruption_clusters_path)
    plt.close()
    return corruption_clusters_path

def preprocess_text(text):
    return text.lower()

def similarity_heatmap():
    df = pd.read_csv("comments.csv")
    df['comments'] = df['comments'].apply(preprocess_text)
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['comments'])
    cosine_sim = cosine_similarity(tfidf_matrix)
    
    # Кластеризация
    num_clusters = 5
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(tfidf_matrix)
    clusters = kmeans.labels_
    
    cluster_sim = np.zeros((num_clusters, num_clusters))
    for i in range(num_clusters):
        for j in range(num_clusters):
            cluster_sim[i, j] = np.mean(cosine_sim[clusters == i][:, clusters == j])
    
    # Создание тепловой карты
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cluster_sim, cmap='viridis', annot=True, fmt=".2f", cbar=True)
    
    # Инвертирование оси y
    ax.invert_yaxis()
    
    plt.title('Тепловая карта схожести кластеров')
    
    heatmap_path = os.path.join('static', 'similarity_heatmap.png')
    plt.savefig(heatmap_path)
    plt.close()
    return heatmap_path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/map')
def map_page():
    return render_template('map.html')

@app.route('/parse_comments', methods=['POST'])
def parse_comments():
    source = request.form['source']
    urls = request.form['urls'].split(',')
    comments = []
    likes = []

    if source == 'youtube':
        for url in urls:
            video_comments, video_likes = get_comments(url)
            comments.extend(video_comments)
            likes.extend(video_likes)
    elif source == 'habr':
        for url in urls:
            habr_comments = get_habr_comments(url)
            comments.extend([comment['comments'] for comment in habr_comments])
            likes.extend([comment['likes'] for comment in habr_comments])
    elif source == 'reddit':
        reddit_credentials = load_reddit_credentials()
        for url in urls:
            reddit_comments = get_reddit_comments(url, reddit_credentials)
            comments.extend([comment['comments'] for comment in reddit_comments])
            likes.extend([comment['likes'] for comment in reddit_comments])
    elif source == 'steam':
        steam_comments_df = get_steam_comments(urls)
        comments = steam_comments_df['comments'].tolist()
        likes = steam_comments_df['likes'].tolist()
    else:
        return "Invalid source", 400

    comments_data = pd.DataFrame({
        'comments': comments,
        'likes': likes
    })
    if 'sentiment' not in comments_data.columns:
        comments_data['sentiment'] = comments_data['comments'].apply(lambda x: model(x[:512])[0]['label'])  # Обрезаем комментарий до 512 токенов
    comments_data.to_csv('comments.csv', index=False, encoding='utf-8')
    combined_data = list(zip(comments, likes))
    return render_template('results.html', data=combined_data)

@app.route('/analyze_comments', methods=['POST'])
def analyze_comments_route():
    action = request.form['action']
    
    if action == 'wordcloud':
        return redirect(url_for('wordcloud_page'))
    elif action == 'sentiment':
        return redirect(url_for('sentiment_analysis'))
    elif action == 'max_likes':
        max_likes()
        return redirect(url_for('display_top_liked_comments'))
    elif action == 'histogram':
        return redirect(url_for('histogram'))
    else:
        return "Invalid action", 400

@app.route('/sentiment_analysis')
def sentiment_analysis():
    result_df = analyze_comments()
    return render_template('sentiment_analysis.html', tables=result_df.to_dict(orient='records'))

@app.route('/clusters')
def clusters():
    clusters_df = pd.read_csv('clusters.csv')
    clusters_df['Комментарий'] = clusters_df['Комментарий'].apply(lambda x: x[:200] + '...' if len(x) > 200 else x)
    return render_template('clusters.html', tables=[clusters_df.to_html(classes='data', header="true", index=False)], titles=['Кластеры'])

@app.route('/wordcloud')
def wordcloud_page():
    wordcloud_image = create_wordcloud_from_comments()
    image = wordcloud_image.to_image()
    img_io = io.BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
    return render_template('wordcloud.html', image_data=img_base64)

@app.route('/top_liked_comments')
def display_top_liked_comments():
    try:
        df = pd.read_csv('top_liked_comments.csv')
    except FileNotFoundError:
        return "Файл с наиболее понравившимися комментариями не найден. Пожалуйста, убедитесь, что он был сгенерирован сначала.", 404
    except Exception as e:
        return f"Произошла ошибка: {e}", 500
    return render_template('top_liked_comments.html', tables=[df.to_html(classes='data', header="true", index=False)], titles=['Топ 50 понравившихся комментариев'])

@app.route('/histogram')
def histogram():
    histogram_path = generate_histogram()
    with open(histogram_path, "rb") as f:
        image = f.read()
    return render_template('histogram.html', image_data=base64.b64encode(image).decode('utf-8'))

@app.route('/search_word', methods=['POST'])
def search_word_route():
    word = request.form['word']
    matched_comments_df = search_word_in_comments(word)
    matched_comments_df = matched_comments_df.to_dict(orient='records')
    return render_template('search_word.html', matched_comments=matched_comments_df, titles=['Результаты поиска'])

@app.route('/sentiment_distribution')
def sentiment_distribution_route():
    distribution_path = sentiment_distribution()
    if isinstance(distribution_path, tuple):
        return distribution_path  # Handle the error message
    with open(distribution_path, "rb") as f:
        image = f.read()
    return render_template('sentiment_distribution.html', image_data=base64.b64encode(image).decode('utf-8'))

@app.route('/comment_length_distribution')
def comment_length_distribution_route():
    length_distribution_path = comment_length_distribution()
    if isinstance(length_distribution_path, tuple):
        return length_distribution_path  # Handle the error message
    with open(length_distribution_path, "rb") as f:
        image = f.read()
    return render_template('comment_length_distribution.html', image_data=base64.b64encode(image).decode('utf-8'))

@app.route('/comment_length_stats')
def comment_length_stats_route():
    stats_df = comment_length_stats()
    if isinstance(stats_df, tuple):
        return stats_df  # Handle the error message
    return render_template('comment_length_stats.html', tables=[stats_df.to_html(classes='data')], titles=stats_df.columns.values)

@app.route('/bribe_comment_count')
def bribe_comment_count_route():
    bribe_count_path = bribe_comment_count()
    with open(bribe_count_path, "rb") as f:
        image = f.read()
    return render_template('bribe_comment_count.html', image_data=base64.b64encode(image).decode('utf-8'))

@app.route('/corruption_clusters')
def corruption_clusters_route():
    corruption_clusters_path = corruption_clusters()
    with open(corruption_clusters_path, "rb") as f:
        image = f.read()
    return render_template('corruption_clusters.html', image_data=base64.b64encode(image).decode('utf-8'))

@app.route('/generate_heatmap')
def generate_heatmap():
    heatmap_path = create_heatmap()
    return redirect(url_for('city_heatmap'))

@app.route('/cluster_heatmap')
def heatmap_route():
    heatmap_path = similarity_heatmap()
    with open(heatmap_path, "rb") as f:
        image = f.read()
    return render_template('similarity_heatmap.html', image_data=base64.b64encode(image).decode('utf-8'))

@app.route('/city_heatmap')
def city_heatmap():
    heatmap_path = create_heatmap()
    return render_template('heatmap.html')


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    app.run(debug=True, port=5001)    