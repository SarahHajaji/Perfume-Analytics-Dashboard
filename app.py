import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

st.set_page_config(
    
    page_title="Perfume Analytics Dashboard",
    page_icon="perfume_icon.ico",
    layout="wide"
)
st.sidebar.title("🌸 Perfume Analytics")

st.sidebar.markdown("""
Welcome!

Use the filters below to explore
different perfume characteristics.
""")

st.sidebar.divider()

df = pd.read_csv("clean_perfume_data.csv")

# ===========================
# Dashboard Header
# ===========================

col1, col2 = st.columns([1, 6])

with col1:
    st.image("perfume_icon.ico", width=80)

with col2:
    st.title("Perfume Analytics Dashboard")
    st.markdown(
        "<span style='color:gray;'>Interactive Dashboard for Exploring Perfume Characteristics, Ratings and User Preferences</span>",
        unsafe_allow_html=True
    )

st.markdown("---")

st.markdown("""
### 👋 Welcome!

This dashboard provides an interactive analysis of perfumes,
including their characteristics, ratings, performance,
user popularity, brand trends, and machine learning predictions.

Use the filters in the sidebar to customize the analysis.
""")



brand = st.sidebar.selectbox(
    "Brand",
    ["All"] + sorted(df["brand"].dropna().unique().tolist())
)

gender = st.sidebar.multiselect(
    "Gender",
    options=df["gender"].dropna().unique(),
    default=df["gender"].dropna().unique()
)

valid_years = df[df["year"] >= 1990]

year = st.sidebar.slider(
    "Year",
    int(valid_years["year"].min()),
    int(valid_years["year"].max()),
    (
        int(valid_years["year"].min()),
        int(valid_years["year"].max())
    )
)


filtered_df = df.copy()

filtered_df = filtered_df[
    filtered_df["gender"].isin(gender)
]

filtered_df = df.copy()

filtered_df = filtered_df[
    filtered_df["year"].between(year[0], year[1])
]

if brand != "All":
    filtered_df = filtered_df[
        filtered_df["brand"] == brand
    ]

st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Perfumes",
    len(filtered_df)
)

col2.metric(
    "Brands",
    filtered_df["brand"].nunique()
)

col3.metric(
    "Years",
    filtered_df["year"].nunique()
)    

st.subheader("Data Preview")

st.dataframe(filtered_df.head())

st.subheader("Summary Statistics")

st.dataframe(filtered_df.describe())

st.divider()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🌸 Characteristics",
    "⭐ Ratings",
    "✨ Performance",
    "❤️ User Interaction",
    "🏷️ Brands",
    "📈 Correlation",
    "🤖 Prediction"
])

with tab1:

    st.header("🌸 Perfume Characteristics Analysis")

    st.write("""
    This section explores the main characteristics of perfumes,
    including gender category, seasonal suitability,
    day or night use, and the most common fragrance notes.
    """)

    # =========================================================
    # Question 1
    # =========================================================

    st.subheader("❓ What types of perfumes are included in the dataset?")

    gender_count = (
        filtered_df["gender"]
        .value_counts()
        .reset_index()
    )

    gender_count.columns = ["Gender", "Count"]

    fig = px.bar(
        gender_count,
        x="Gender",
        y="Count",
        color="Count",
        text="Count",
        title="Types of Perfumes by Gender"
    )

    fig.update_layout(template="plotly_white")

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
The chart shows that perfumes are distributed across different gender categories,
with some categories being more common than others.
This suggests that fragrance brands tend to focus more on certain target audiences.
""")

    st.divider()

    # =========================================================
    # Question 2
    # =========================================================

    st.subheader("❓ Which season are perfumes most suitable for?")

    season = pd.DataFrame({

        "Season":[
            "Winter",
            "Spring",
            "Summer",
            "Autumn"
        ],

        "Votes":[
            filtered_df["winter"].sum(),
            filtered_df["spring"].sum(),
            filtered_df["summer"].sum(),
            filtered_df["autumn"].sum()
        ]

    })

    fig = px.bar(

        season,

        x="Season",

        y="Votes",

        color="Votes",

        text="Votes",

        title="Seasonal Suitability of Perfumes"

    )

    fig.update_layout(template="plotly_white")

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
Perfumes receive different levels of suitability across the four seasons.
Overall, warmer and milder seasons tend to receive more votes than colder seasons,
suggesting that many perfumes are designed for everyday weather conditions.
""")

    st.divider()

    # =========================================================
    # Question 3
    # =========================================================

    st.subheader("❓ Are perfumes more suitable for daytime or nighttime wear?")

    time = pd.DataFrame({

        "Type":[
            "Day",
            "Night"
        ],

        "Votes":[
            filtered_df["day"].sum(),
            filtered_df["night"].sum()
        ]

    })

    fig = px.pie(

        time,

        values="Votes",

        names="Type",

        hole=0.45,

        title="Day vs Night Perfumes"

    )

    fig.update_layout(template="plotly_white")

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
The chart shows that daytime perfumes are more common than nighttime perfumes.
This suggests that many fragrances are designed for everyday use
rather than special evening occasions.
""")

    st.divider()

    # =========================================================
    # Question 4
    # =========================================================

    st.subheader("❓ How do the most common top notes differ between winter and summer perfumes?")

    winter_perfumes = filtered_df[
        filtered_df["winter"] >= filtered_df["winter"].quantile(0.75)
    ]

    summer_perfumes = filtered_df[
        filtered_df["summer"] >= filtered_df["summer"].quantile(0.75)
    ]

    winter_notes = (
        winter_perfumes["notes_top"]
        .dropna()
        .str.split("|", regex=False)
        .explode()
    )

    summer_notes = (
        summer_perfumes["notes_top"]
        .dropna()
        .str.split("|", regex=False)
        .explode()
    )

    winter_counts = winter_notes.value_counts()
    summer_counts = summer_notes.value_counts()

    notes_df = pd.DataFrame({
        "Winter": winter_counts,
        "Summer": summer_counts
    }).fillna(0)

    notes_df["Total"] = notes_df["Winter"] + notes_df["Summer"]

    notes_df = (
        notes_df
        .sort_values("Total", ascending=False)
        .head(10)
        .drop(columns="Total")
        .reset_index()
    )

    notes_df.columns = ["Top Note", "Winter", "Summer"]

    notes_long = notes_df.melt(
        id_vars="Top Note",
        value_vars=["Winter", "Summer"],
        var_name="Season",
        value_name="Count"
    )

    fig = px.bar(
        notes_long,
        x="Top Note",
        y="Count",
        color="Season",
        barmode="group",
        title="Most Common Top Notes in Winter and Summer"
    )

    fig.update_layout(template="plotly_white")

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
Winter and summer perfumes share some popular top notes,
especially Bergamot.

However, citrus notes such as Lemon and Grapefruit appear
more often in summer perfumes, while warmer spicy notes like
Cardamom and Pink Pepper are more common in winter perfumes.
""")
with tab2:

    st.header("⭐ Rating Analysis")

    st.write("""
    This section explores how users rate perfumes by analyzing
    rating distributions, user votes, and the relationship
    between ratings and other factors.
    """)

    # =========================================================
    # Question 1
    # =========================================================

    st.subheader("❓ How are perfume ratings distributed?")

    fig = px.histogram(

        filtered_df[
            filtered_df["average_rating"] > 0
        ],

        x="average_rating",

        nbins=20,

        title="Distribution of Average Ratings"

    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Average Rating",
        yaxis_title="Frequency"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
Most perfumes receive relatively high average ratings,
indicating that users generally have positive opinions
about the perfumes included in the dataset.
""")

    st.divider()

    # =========================================================
    # Question 2
    # =========================================================

    st.subheader("❓ Do perfumes with more user votes receive higher average ratings?")

    fig = px.scatter(

        filtered_df,

        x="users_rated",

        y="average_rating",

        opacity=0.5,

        color="average_rating",

        title="Average Rating vs Number of User Ratings"

    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Number of User Ratings",
        yaxis_title="Average Rating"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
Perfumes with a large number of votes tend to maintain
ratings around 4.0.

However, highly rated perfumes can also have relatively
few votes, suggesting that popularity and rating are
related but not strongly dependent on each other.
""")

    st.divider()

    # =========================================================
    # Question 3
    # =========================================================

    st.subheader("❓ Are there perfumes that receive exceptionally high numbers of user votes?")

    fig = px.box(

        filtered_df,

        x="users_rated",

        title="Distribution of Total Votes"

    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Total Votes"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
Most perfumes receive a relatively small number of user ratings,
while a few perfumes appear as clear outliers with exceptionally
high numbers of votes.

This suggests that user attention is concentrated on a limited
number of fragrances.
""")

    st.divider()

    # =========================================================
    # Question 4
    # =========================================================

    st.subheader("❓ How do users usually rate perfumes?")

    rating_distribution = pd.DataFrame({

        "Rating":[
            "1 Star",
            "2 Stars",
            "3 Stars",
            "4 Stars",
            "5 Stars"
        ],

        "Count":[

            filtered_df["rating_1_star"].sum(),
            filtered_df["rating_2_stars"].sum(),
            filtered_df["rating_3_stars"].sum(),
            filtered_df["rating_4_stars"].sum(),
            filtered_df["rating_5_stars"].sum()

        ]

    })

    fig = px.bar(

        rating_distribution,

        x="Rating",

        y="Count",

        color="Count",

        text="Count",

        title="Distribution of User Ratings"

    )

    fig.update_layout(template="plotly_white")

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
Most users are happy with the perfumes they try.

The majority of ratings are 4 or 5 stars,
while 1-star ratings are very rare.

Overall, users tend to leave positive reviews
rather than negative ones.
""")

with tab3:

    st.header("🌿 Fragrance Performance Analysis")

    st.write("""
    This section explores how users evaluate perfume performance,
    including longevity, sillage, and value for money.
    """)

    # =========================================================
    # Question 1
    # =========================================================

    st.subheader("❓ Do perfumes with better longevity receive higher ratings?")

    fig = px.scatter(

        filtered_df,

        x="average_longevity",

        y="average_rating",

        color="average_rating",

        opacity=0.5,

        title="Average Rating vs Average Longevity"

    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Average Longevity",
        yaxis_title="Average Rating"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
The scatter plot suggests a slight positive relationship
between average longevity and average rating.

Perfumes with longer longevity often receive good ratings,
but longevity alone does not determine how highly
a perfume is rated.
""")

    st.divider()

    # =========================================================
    # Question 2
    # =========================================================

    st.subheader("❓ What sillage level do users prefer most?")

    sillage_distribution = pd.DataFrame({

        "Sillage":[
            "Intimate",
            "Moderate",
            "Strong",
            "Enormous"
        ],

        "Count":[

            filtered_df["intimate"].sum(),
            filtered_df["moderate_sillage"].sum(),
            filtered_df["strong"].sum(),
            filtered_df["enormous"].sum()

        ]

    })

    fig = px.bar(

        sillage_distribution,

        x="Sillage",

        y="Count",

        color="Count",

        text="Count",

        title="Distribution of Sillage Levels"

    )

    fig.update_layout(template="plotly_white")

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
Most perfumes in the dataset fall into the Moderate
sillage category.

This suggests that users generally prefer fragrances
that are noticeable without being too overpowering.
""")

    st.divider()

    # =========================================================
    # Question 3
    # =========================================================

    st.subheader("❓ How do users evaluate the value for money of perfumes?")

    price_distribution = pd.DataFrame({

        "Value":[
            "Overpriced",
            "Poor",
            "Fair",
            "Good",
            "Excellent"
        ],

        "Count":[

            filtered_df["overpriced"].sum(),
            filtered_df["poor_value"].sum(),
            filtered_df["fair_value"].sum(),
            filtered_df["good_value"].sum(),
            filtered_df["excellent_value"].sum()

        ]

    })

    fig = px.bar(

        price_distribution,

        x="Value",

        y="Count",

        color="Count",

        text="Count",

        title="Distribution of Price Value Ratings"

    )

    fig.update_layout(template="plotly_white")

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
Most users think the perfumes offer fair value for money,
and many also rate them as good value.

Only a small number of perfumes are considered overpriced,
giving an overall positive impression of their value.
""")
with tab4:

    st.header("🔥 Popularity Analysis")

    st.write("""
    This section explores how users interact with different perfumes
    by analyzing ownership, previous ownership, and future interest.
    """)

    # =========================================================
    # Question 1
    # =========================================================

    st.subheader("❓ Which perfumes are the most wanted?")

    top_wanted = filtered_df.nlargest(10, "users_want").copy()

    top_wanted["Perfume"] = (
        top_wanted["name"] + " - " + top_wanted["brand"]
    )

    fig = px.bar(

        top_wanted,

        x="users_want",

        y="Perfume",

        orientation="h",

        color="users_want",

        title="Top 10 Most Wanted Perfumes"

    )

    fig.update_layout(template="plotly_white")

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
The chart shows the ten perfumes with the highest user demand.
These fragrances continue to attract a large number of users,
highlighting their strong popularity within the community.
""")

    st.divider()

    # =========================================================
    # Question 2
    # =========================================================

    st.subheader("❓ Which perfumes are owned by the largest number of users?")

    top_have = filtered_df.nlargest(10, "users_have").copy()

    top_have["Perfume"] = (
        top_have["name"] + " - " + top_have["brand"]
    )

    fig = px.bar(

        top_have,

        x="users_have",

        y="Perfume",

        orientation="h",

        color="users_have",

        title="Top 10 Most Owned Perfumes"

    )

    fig.update_layout(template="plotly_white")

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
These perfumes have the largest number of current owners,
indicating that they are among the most established
and widely used fragrances in the dataset.
""")

    st.divider()

    # =========================================================
    # Question 3
    # =========================================================

    st.subheader("❓ Which perfumes were previously owned by the largest number of users?")

    top_had = filtered_df.nlargest(10, "users_had").copy()

    top_had["Perfume"] = (
        top_had["name"] + " - " + top_had["brand"]
    )

    fig = px.bar(

        top_had,

        x="users_had",

        y="Perfume",

        orientation="h",

        color="users_had",

        title="Top 10 Most Previously Owned Perfumes"

    )

    fig.update_layout(template="plotly_white")

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
These perfumes were previously owned by many users,
suggesting that they have remained well known over time
and have reached a large audience.
""")

    st.divider()

    # =========================================================
    # Question 4
    # =========================================================

    st.subheader("❓ How do ownership and user interest compare for the most popular perfumes?")

    popularity = filtered_df.copy()

    popularity["total_popularity"] = (
        popularity["users_have"] +
        popularity["users_had"] +
        popularity["users_want"]
    )

    top_perfumes = popularity.nlargest(10, "total_popularity").copy()

    top_perfumes["Perfume"] = (
        top_perfumes["brand"] + " - " + top_perfumes["name"]
    )

    comparison = top_perfumes[
        ["Perfume", "users_have", "users_had", "users_want"]
    ]

    comparison = comparison.melt(
        id_vars="Perfume",
        var_name="Status",
        value_name="Users"
    )

    comparison["Status"] = comparison["Status"].replace({
        "users_have": "Have",
        "users_had": "Had",
        "users_want": "Want"
    })

    fig = px.bar(

        comparison,

        x="Users",

        y="Perfume",

        color="Status",

        barmode="group",

        title="Ownership and User Interest Comparison"

    )

    fig.update_layout(template="plotly_white")

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
Most of the top perfumes have a high number of users
who currently own them.

At the same time, many users still want these perfumes,
showing that they continue to be popular even after
becoming widely owned.
""")
with tab5:

    st.header("🏷️ Brand & Trend Analysis")

    st.write("""
    This section explores perfume brands, release trends,
    and the highest-rated brands in the dataset.
    """)

    # =========================================================
    # Question 1
    # =========================================================

    st.subheader("❓ How has the number of perfume releases changed over time?")

    df_year = filtered_df[
        (filtered_df["year"] >= 1900) &
        (filtered_df["year"] <= 2025)
    ]

    perfumes_year = (
        df_year
        .groupby("year")
        .size()
        .reset_index(name="Count")
    )

    fig = px.line(
        perfumes_year,
        x="year",
        y="Count",
        markers=True,
        title="Perfumes Released by Year"
    )

    fig.update_layout(template="plotly_white")

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
The number of perfume releases has generally increased over time,
especially during recent decades.

This indicates continuous growth in the fragrance industry
and the introduction of many new products each year.
""")

    st.divider()

    # =========================================================
    # Question 2
    # =========================================================

    st.subheader("❓ Which brands have released the largest number of perfumes?")

    top_brands = (
        filtered_df["brand"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    top_brands.columns = ["Brand", "Count"]

    fig = px.bar(
        top_brands,
        x="Count",
        y="Brand",
        orientation="h",
        color="Count",
        text="Count",
        title="Top 10 Brands by Number of Perfumes"
    )

    fig.update_layout(template="plotly_white")

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
A small number of brands dominate the dataset
by releasing significantly more perfumes than others.

These brands have built extensive fragrance collections
over many years.
""")

    st.divider()

    # =========================================================
    # Question 3
    # =========================================================

    st.subheader("❓ Which brands have the highest average ratings?")

    brand_rating = (
        filtered_df
        .groupby("brand")
        .agg(
            Average_Rating=("average_rating", "mean"),
            Number_of_Perfumes=("name", "count")
        )
    )

    brand_rating = brand_rating[
        brand_rating["Number_of_Perfumes"] >= 10
    ]

    top_rating = (
        brand_rating
        .sort_values(
            "Average_Rating",
            ascending=False
        )
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        top_rating,
        x="Average_Rating",
        y="brand",
        orientation="h",
        color="Average_Rating",
        text_auto=".2f",
        title="Top 10 Highest Rated Brands"
    )

    fig.update_layout(template="plotly_white")

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
Several brands consistently receive excellent average ratings.

Restricting the analysis to brands with at least
10 perfumes provides a fair comparison and avoids
bias from brands with only one or two highly rated perfumes.
""")
with tab6:

    st.header("📈 Correlation Analysis")

    st.write("""
    This section examines the relationships between the main numerical
    features to identify which perfume characteristics are related.
    """)

    # =========================================================
    # Question 1
    # =========================================================

    st.subheader("❓ What relationships exist between the main numerical features?")

    corr_columns = [

        "average_rating",
        "users_rated",
        "average_longevity",
        "average_sillage",
        "average_price_value",
        "users_have",
        "users_had",
        "users_want"

    ]

    corr = filtered_df[corr_columns].corr()

    fig = px.imshow(

        corr,

        text_auto=".2f",

        color_continuous_scale="RdBu_r",

        aspect="auto",

        title="Correlation Heatmap"

    )

    fig.update_layout(template="plotly_white")

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
Most relationships between variables are relatively weak.

However, user interaction variables such as ownership,
previous ownership, and user interest show positive
correlations because popular perfumes tend to receive
more attention from users.
""")

    st.divider()

    # =========================================================
    # Question 2
    # =========================================================

    st.subheader("❓ How are the main performance features related to each other?")

    performance_corr = filtered_df[

        [

            "average_rating",
            "average_longevity",
            "average_sillage",
            "average_price_value"

        ]

    ].corr()

    fig = px.imshow(

        performance_corr,

        text_auto=".2f",

        color_continuous_scale="Blues",

        aspect="auto",

        title="Performance Features Correlation"

    )

    fig.update_layout(template="plotly_white")

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
Average rating has a slight positive relationship with
longevity, sillage, and value for money.

Although these characteristics contribute to users'
overall satisfaction, none of them alone strongly
determines the final perfume rating.
""")
with tab7:

    rating_model = joblib.load("rating_model.pkl")

    st.header("🤖 Perfume Rating Prediction")

    st.write("""
    Predict the expected average perfume rating based on its characteristics.
    """)

    col1, col2 = st.columns(2)

    with col1:

        longevity = st.number_input(
            "Average Longevity",
            min_value=0.0,
            max_value=5.0,
            value=3.0,
            step=0.1
        )

        sillage = st.number_input(
            "Average Sillage",
            min_value=0.0,
            max_value=4.0,
            value=2.0,
            step=0.1
        )

    with col2:

        price = st.number_input(
            "Average Price Value",
            min_value=0.0,
            max_value=5.0,
            value=3.0,
            step=0.1
        )

        year = st.number_input(
            "Year",
            min_value=1990,
            max_value=2030,
            value=2020
        )

    if st.button("⭐ Predict Rating"):

        sample = [[
            longevity,
            sillage,
            price,
            year
        ]]

        prediction = rating_model.predict(sample)[0]

        st.success(f"⭐ Predicted Rating: {prediction:.2f}")

    st.divider()

    st.header("🌸 Perfume Recommendation")

    st.write("""
    Find perfumes that match your preferences.
    """)

    col1, col2 = st.columns(2)

    with col1:

        gender_filter = st.selectbox(
            "Gender",
            sorted(df["gender"].dropna().unique())
        )

        season_filter = st.selectbox(
            "Season",
            ["Winter", "Spring", "Summer", "Autumn"]
        )

    with col2:

        time_filter = st.selectbox(
            "Day / Night",
            ["Day", "Night"]
        )

        min_rating = st.slider(
            "Minimum Rating",
            1.0,
            5.0,
            4.0,
            0.1
        )

    if st.button("🌸 Find My Perfume"):

        recommendation = df.copy()

        recommendation = recommendation[
            recommendation["gender"] == gender_filter
        ]

        recommendation = recommendation[
            recommendation[season_filter.lower()] > 0
        ]

        recommendation = recommendation[
            recommendation[time_filter.lower()] > 0
        ]

        recommendation = recommendation[
            recommendation["average_rating"] >= min_rating
        ]

        recommendation = recommendation.sort_values(
            by=["average_rating", "users_rated"],
            ascending=[False, False]
        ).head(5)

        if recommendation.empty:

            st.warning("No perfumes match your preferences.")

        else:

            st.success("Top Recommended Perfumes")

            for _, row in recommendation.iterrows():

                st.markdown(f"""
### 🧴 {row['name']}

**🏷️ Brand:** {row['brand']}

⭐ **Rating:** {row['average_rating']:.2f}

👥 **Users Rated:** {int(row['users_rated']):,}

📅 **Year:** {int(row['year'])}

---
""")
st.markdown(
    """
    <div style='text-align:center; padding:20px;'>

    <h3>🌸 Perfume Analytics Dashboard</h3>

    <p>Developed by <b>Sarah Ahmed Hajjaji</b></p>


    </div>
    """,
    unsafe_allow_html=True
)