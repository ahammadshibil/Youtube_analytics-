import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime
from wordcloud import WordCloud
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import pycountry
import plotly.figure_factory as ff


# Define YouTube color palette
yt_red = '#FF0000'
bg_color = '#FFFFFF'
text_color = '#282828'


# CSS for Streamlit theme
theme_css = """
<style>
    body {
        background-color: #FFFFFF;
    }
    h1, h2, h3, h4 {
        color: #FF0000;
    }
    div[role="listbox"] {
        background-color: #FFFFFF;
    }
</style>
"""

# Inject the CSS into Streamlit
st.markdown(theme_css, unsafe_allow_html=True)

# Your previous JS code
js_code = """
<script>
    setTimeout(function() {
        var bgColor = window.getComputedStyle(document.querySelector("div[role='main']")).backgroundColor;
        if (bgColor === "rgb(255, 255, 255)") {
            // Light theme detected, set text to dark color
            document.querySelector("div[role='main']").style.color = "black";
            document.querySelectorAll("th, .dynamic-theme td").forEach(elem => elem.style.color = "black");
        } else {
            // Dark theme detected, set text to light color
            document.querySelector("div[role='main']").style.color = "white";
            document.querySelectorAll("th, .dynamic-theme td").forEach(elem => elem.style.color = "white");
        }
    }, 100);
</script>
"""

st.markdown(js_code, unsafe_allow_html=True)


# Your app code here
st.title("YouTube Analytics Dashboard")
st.write("This dashboard adjusts dynamically to Streamlit's theme.")



def audience_simple(country):
    """Show top represented countries"""
    if country == 'US':
        return 'USA'
    elif country == 'IN':
        return 'India'
    else:
        return 'Other'

def str_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s


def style_negative(val):
    """
    Takes a scalar and returns a string with
    the CSS property `'color: red'` if value is negative, black otherwise.
    """
    color = 'red' if (isinstance(val, (int, float)) and val < 0) else 'black'
    return 'color: %s' % color

def style_positive(val):
    """
    Takes a scalar and returns a string with
    the CSS property `'color: green'` if value is positive, black otherwise.
    """
    color = 'green' if (isinstance(val, (int, float)) and val > 0) else 'black'
    return 'color: %s' % color

def country_code_to_name(code):
    try:
        return pycountry.countries.get(alpha_2=code).name
    except (AttributeError, LookupError):
        return 'Other'
    
def plot_histogram(data, metric, position, title):
    plt.subplot(5, 2, position)
    sns.histplot(data[metric], color="#FF0000", bins=30, kde=False)
    plt.title(title)
    plt.ylabel("Amount of Videos")
    plt.xlabel(metric)


def load_data():
    # Read CSV files
    df_agg = pd.read_csv('/home/shibil/Annual report/Streamlit/Aggregated_Metrics_By_Video.csv').iloc[1:,:]
    df_agg_sub = pd.read_csv('/home/shibil/Annual report/Streamlit/Aggregated_Metrics_By_Country_And_Subscriber_Status.csv')
    df_comments = pd.read_csv('/home/shibil/Annual report/Streamlit/Aggregated_Metrics_By_Video.csv')
    df_all_comments = pd.read_csv('/home/shibil/Annual report/Streamlit/All_Comments_Final.csv')
    df_time = pd.read_csv('/home/shibil/Annual report/Streamlit/Video_Performance_Over_Time.csv')
    
    # Rename columns
    df_agg.columns = ['Video', 'Video title', 'Video publish time', 'Comments added', 'Shares', 'Dislikes', 'Likes', 
                  'Subscribers lost', 'Subscribers gained', 'RPM(USD)', 'CPM(USD)', 'Average % viewed', 
                  'Average view duration', 'Views', 'Watch time (hours)', 'Subscribers', 
                  'Your estimated revenue (USD)', 'Impressions', 'Impressions ctr(%)']

    
    df_all_comments.rename(columns={'VidId': 'Video', 'Comments': 'comment_text'}, inplace=True)
    df_all_comments['sentiment'] = df_all_comments['comment_text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    # Convert columns to datetime format and perform calculations
    df_agg['Video publish time'] = pd.to_datetime(df_agg['Video publish time'], format="%b %d, %Y")
    df_agg['Average view duration in seconds'] = df_agg['Average view duration'].apply(str_to_seconds)
    df_agg['Engagement_ratio'] =  (df_agg['Comments added'] + df_agg['Shares'] +df_agg['Dislikes'] + df_agg['Likes']) /df_agg.Views
    df_agg['Views / sub gained'] = df_agg['Views'] / df_agg['Subscribers gained']
    df_agg['Average view duration'] = df_agg['Average view duration'].astype(str).apply(lambda x: datetime.strptime(x, '%H:%M:%S') if x != 'nan' else x)
    df_agg['Avg_duration_sec'] = df_agg['Average view duration'].apply(lambda x: x.second + x.minute * 60 + x.hour * 3600 if x != 'nan' else 0)
    df_time['Date'] = pd.to_datetime(df_time['Date'], errors='coerce')

    return df_agg, df_agg_sub, df_comments, df_time, df_all_comments

df_agg, df_agg_sub, df_comments, df_time, df_all_comments = load_data()

    
#print(df_time['Date'])
df_agg_diff = df_agg.copy()
metric_date_12mo = df_agg_diff['Video publish time'].max() - pd.DateOffset(months=12)

# Filter to rows based on 'Video publish time' condition
filtered_df = df_agg_diff[df_agg_diff['Video publish time'] >= metric_date_12mo]

# Only include numeric columns
numeric_cols = filtered_df.select_dtypes(include=[np.number])

# Calculate the median
median_agg = numeric_cols.median(numeric_only=True)
df_agg['Engagement Index'] = df_agg['Likes'] + df_agg['Shares'] + df_agg['Comments added']


#print(numeric_cols )
#print(filtered_df)


numeric_cols = np.array((df_agg_diff.dtypes == 'float64') | (df_agg_diff.dtypes == 'int64'))
df_agg_diff.iloc[:,numeric_cols] = (df_agg_diff.iloc[:,numeric_cols] - median_agg).div(median_agg)


df_time_diff = pd.merge(df_time, df_agg.loc[:,['Video','Video publish time']], left_on ='External Video ID', right_on = 'Video')
df_time_diff['days_published'] = (df_time_diff['Date'] - df_time_diff['Video publish time']).dt.days

date_12mo = df_agg['Video publish time'].max() - pd.DateOffset(months =12)
df_time_diff_yr = df_time_diff[df_time_diff['Video publish time'] >= date_12mo]
views_days = pd.pivot_table(df_time_diff_yr,index= 'days_published',values ='Views', aggfunc = [np.mean,np.median,lambda x: np.percentile(x, 80),lambda x: np.percentile(x, 20)]).reset_index()
views_days.columns = ['days_published','mean_views','median_views','80pct_views','20pct_views']
views_days = views_days[views_days['days_published'].between(0,30)]
views_cumulative = views_days.loc[:,['days_published','median_views','80pct_views','20pct_views']] 
views_cumulative.loc[:,['median_views','80pct_views','20pct_views']] = views_cumulative.loc[:,['median_views','80pct_views','20pct_views']].cumsum()
#print(df_agg_diff['Video publish time'].apply(type).value_counts())
add_sidebar = st.sidebar.selectbox('Aggregate or Individual Video', ('Aggregate Metrics','Individual Video Analysis'))
#print(df_agg_diff.columns)


# Assuming country_metrics is already loaded

# Image URLs for top 10 videos
img_name = [
    "https://i.ytimg.com/vi/4OZip0cgOho/hqdefault.jpg",
    "https://i.ytimg.com/vi/Ip50cXvpWY4/hqdefault.jpg",
    "https://i.ytimg.com/vi/8igH8qZafpo/hqdefault.jpg",
    "https://i.ytimg.com/vi/I3FBJdiExcg/hqdefault.jpg",
    "https://i.ytimg.com/vi/yukdXV9LR48/hqdefault.jpg",
    "https://i.ytimg.com/vi/41Clrh6nv1s/hqdefault.jpg",
    "https://i.ytimg.com/vi/sHRq-LshG3U/hqdefault.jpg",
    "https://i.ytimg.com/vi/MpF9HENQjDo/hqdefault.jpg",
    "https://i.ytimg.com/vi/SVtRsDhHlDk/hqdefault.jpg",
    "https://i.ytimg.com/vi/m5pwx3hgtzM/hqdefault.jpg"
]

# Grouping and creating the thumbnail_df
thumbnail_df = df_agg_sub.groupby(['Thumbnail link', 'Video Title', 'Video Length']).agg({
    'Views': 'sum',
    'Video Likes Added': 'sum',
    'Video Dislikes Added': 'sum',
    'Average View Percentage': 'mean',
    'Average Watch Time': 'mean',
    'User Comments Added': 'sum'
}).reset_index()

# Sorting and taking the top 10 based on views
thumbnail_df = thumbnail_df.sort_values(by='Views', ascending=False).head(10)

# Adding the img_name to the dataframe and computing the ratio
thumbnail_df['img'] = img_name
thumbnail_df['ratio'] = (thumbnail_df['Video Likes Added'] - thumbnail_df['Video Dislikes Added']) / (thumbnail_df['Video Likes Added'] + thumbnail_df['Video Dislikes Added'])
thumbnail_df['ratio'] = thumbnail_df['ratio'] * 100  # Convert to percentage

# Format video length in minutes and seconds
thumbnail_df['Video Length'] = thumbnail_df['Video Length'] // 60 + (thumbnail_df['Video Length'] % 60) / 100

# Format average watch time in minutes and seconds
thumbnail_df['Average Watch Time'] = thumbnail_df['Average Watch Time'] // 60 + (thumbnail_df['Average Watch Time'] % 60) / 100

# Compute average watch time ratio
thumbnail_df['avg_wt_ratio'] = (thumbnail_df['Average Watch Time'] / thumbnail_df['Video Length']) * 100





#Show individual metrics 
if add_sidebar == 'Aggregate Metrics':
    st.write("Ken Jee YouTube Aggregated Data")
    
    df_agg_metrics = df_agg[['Video publish time', 'Views', 'Likes', 'Subscribers', 'Shares', 'Comments added', 'RPM(USD)', 'Average % viewed',
                             'Avg_duration_sec', 'Engagement_ratio', 'Views / sub gained']]
    metric_date_6mo = df_agg_metrics['Video publish time'].max() - pd.DateOffset(months=6)
    metric_date_12mo = df_agg_metrics['Video publish time'].max() - pd.DateOffset(months=12)
    metric_medians6mo = df_agg_metrics[df_agg_metrics['Video publish time'] >= metric_date_6mo].drop(columns=['Video publish time']).median()
    metric_medians12mo = df_agg_metrics[df_agg_metrics['Video publish time'] >= metric_date_12mo].drop(columns=['Video publish time']).median()

    plt.figure(figsize=(20, 30))
    metrics_to_plot = ['Views', 'Comments added', 'Likes', 'Dislikes', 'CPM(USD)', 'RPM(USD)', 'Your estimated revenue (USD)', 
                       'Impressions', 'Impressions ctr(%)', 'Average % viewed']
    titles = [f"Distribution of {metric} for all Videos" for metric in metrics_to_plot]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    columns = [col1, col2, col3, col4, col5]
    #print(type(metric_date_6mo))
    #print(type(metric_date_12mo))
    filtered_data = df_agg_metrics[df_agg_metrics['Video publish time'] >= metric_date_6mo]
    #print(filtered_data['Video publish time'].dtypes)
    # Assuming columns is a list of Streamlit columns defined somewhere above
    columns_count = len(columns)
    for idx, metric_name in enumerate(metric_medians6mo.index):
        column = columns[idx % columns_count]
        with column:
            delta_value = (metric_medians6mo[metric_name] - metric_medians12mo[metric_name]) / metric_medians12mo[metric_name]
            formatted_delta = "{:.2%}".format(delta_value)
            st.metric(label=metric_name, value=round(metric_medians6mo[metric_name], 1), delta=formatted_delta)
    
    
    df_agg_diff['Publish_date'] = df_agg_diff['Video publish time'].apply(lambda x: x.date())
    df_agg_diff_final = df_agg_diff.loc[:, ['Video title', 'Publish_date', 'Views', 'Likes', 'Subscribers', 'Shares', 
                                        'Comments added', 'RPM(USD)', 'Average % viewed', 'Avg_duration_sec', 
                                        'Engagement_ratio', 'Views / sub gained']]

# Only filter numeric columns for formatting
    df_agg_numeric_lst = df_agg_diff_final.select_dtypes(include=[np.number]).columns.tolist()

    df_to_pct = {}
    for i in df_agg_numeric_lst:
        df_to_pct[i] = '{:.1%}'.format
        
# Streamlit display with styles and formatting

    st.dataframe(df_agg_diff_final.style.set_table_attributes('class="dynamic-theme"').applymap(style_negative).applymap(style_positive).format(df_to_pct))    

    for i, metric in enumerate(metrics_to_plot):
        plot_histogram(df_agg, metric, i+1, titles[i])

    plt.tight_layout()

    # Increase font size of the title, x-axis, and y-axis
    plt.rcParams["axes.titlesize"] = 20
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14

    st.pyplot()


    # Yearly Views Trend
    df_agg_yearly = df_agg.groupby(df_agg['Video publish time'].dt.year).agg({
        'Views': 'sum',
        'Likes': 'sum',
    }).reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_agg_yearly, x='Video publish time', y='Views')
    plt.title("Yearly Views Trend")
    plt.xlabel("Year")
    plt.ylabel("Total Views")
    st.pyplot()

    # Number of published YouTube videos by year
    video_counts = df_agg['Video publish time'].dt.year.value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=video_counts.index, y=video_counts.values, color="#FF0000")
    plt.title("Number of published YouTube videos by year")
    plt.xlabel("Year")
    plt.ylabel("Amount of Videos")
    st.pyplot()

    # Top 10 Videos by Views
    top_10_videos = df_agg.nlargest(10, 'Views')   
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_10_videos, y='Video title', x='Views')
    plt.title("Top 10 Videos by Views")
    plt.xlabel("Views")
    plt.ylabel("Video Title")
    st.pyplot()

    # Display table with thumbnails
# Display table with thumbnails
# Display table with thumbnails
    for idx, row in thumbnail_df.iterrows():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.image(row['img'], use_column_width=True)
        with col2:
            st.write("Video Title: ", row['Video Title'])
            st.write("Views: ", row['Views'])
        with col3:
            st.write("Likes: ", row['Video Likes Added'])
            st.write("Dislikes: ", row['Video Dislikes Added'])
        with col4:
            st.write("Avg Watch Time: ", row['Average Watch Time'])
            st.write("Ratio: ", row['ratio'])



if add_sidebar == 'Individual Video Analysis':
    videos = tuple(df_agg['Video title'])
    st.write("Individual Video Performance")
    video_select = st.selectbox('Pick a Video:', videos, key='video_select_individual')
    
    agg_filtered = df_agg[df_agg['Video title'] == video_select]
    agg_sub_filtered = df_agg_sub[df_agg_sub['Video Title'] == video_select]
   
    # Convert country codes to country names
    agg_sub_filtered['Country Name'] = agg_sub_filtered['Country Code'].apply(country_code_to_name)

    # Create choropleth map for views by country for the selected video
    map_data = agg_sub_filtered[['Country Name', 'Views']]

    fig_map = px.choropleth(
        map_data, 
        locations='Country Name',
        locationmode='country names',
        color='Views',
        hover_name='Country Name',
        color_continuous_scale="viridis",
        title="Views by Country for " + video_select
    )
    st.plotly_chart(fig_map)
    
    # Group by country name and sum the views
    agg_grouped = agg_sub_filtered.groupby('Country Name').sum().reset_index()

    # Bar plot for subscriber distribution for selected video
    fig_bar = px.bar(agg_grouped, x='Views', y='Country Name', orientation='h', title='Subscriber Distribution for ' + video_select)
    st.plotly_chart(fig_bar)

    agg_time_filtered = df_time_diff[df_time_diff['Video Title'] == video_select]
    first_30 = agg_time_filtered[agg_time_filtered['days_published'].between(0,30)]
    first_30 = first_30.sort_values('days_published')
    
    # Time series comparison for the video's views in the first 30 days
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['20pct_views'],
            mode='lines', name='20th percentile', line=dict(color='purple', dash ='dash')))
    fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['median_views'],
                mode='lines', name='50th percentile', line=dict(color='black', dash ='dash')))
    fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['80pct_views'],
                mode='lines', name='80th percentile', line=dict(color='royalblue', dash ='dash')))
    fig2.add_trace(go.Scatter(x=first_30['days_published'], y=first_30['Views'].cumsum(),
                mode='lines', name='Current Video', line=dict(color='firebrick',width=8)))
    fig2.update_layout(title='View comparison first 30 days', xaxis_title='Days Since Published', yaxis_title='Cumulative views')
    st.plotly_chart(fig2)

    # Word cloud for selected video comments
    selected_video_id = df_agg[df_agg['Video title'] == video_select]['Video'].iloc[0]
    selected_video_comments = df_all_comments[df_all_comments['Video'] == selected_video_id]['comment_text'].tolist()
    
    def color_func(word, *args, **kwargs):
        sentiment = TextBlob(word).sentiment.polarity
        if sentiment > 0:
            return "green"
        elif sentiment < 0:
            return "red"
        else:
            return "blue"

    text = ' '.join(selected_video_comments)
    wordcloud = WordCloud(background_color="white", color_func=color_func).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
    
# Sentiment Analysis
    selected_video_comments_df = df_all_comments[df_all_comments['Video'] == selected_video_id]
    avg_sentiment = selected_video_comments_df['sentiment'].mean()
    st.write(f"Average Sentiment for {video_select}: {avg_sentiment:.2f}")
    
    fig_sentiment = go.Figure(go.Bar(x=[video_select], y=[avg_sentiment], marker_color=['green' if avg_sentiment > 0 else 'red' if avg_sentiment < 0 else 'gray']))
    st.plotly_chart(fig_sentiment)

    
    
    
    