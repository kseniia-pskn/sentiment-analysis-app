from flask import Blueprint, render_template, jsonify, current_app
from flask_login import login_required, current_user
from .models import User, SentimentSnapshot
import matplotlib.pyplot as plt
import io
import base64
import json

main = Blueprint('main', __name__)

# ----- Home Page -----
@main.route('/')
def index():
    if current_user.is_authenticated:
        return render_template('dashboard.html')
    else:
        return render_template('index.html')


# ----- Dashboard -----
@main.route('/dashboard')
@login_required
def dashboard():
    snapshot = SentimentSnapshot.query.filter_by(user_id=current_user.id).order_by(SentimentSnapshot.timestamp.desc()).first()

    sentiment_chart = trend_chart = country_chart = None

    if snapshot:
        try:
            # Sentiment Breakdown
            fig, ax = plt.subplots()
            ax.bar(['Positive', 'Negative', 'Neutral'],
                   [snapshot.positive_percentage, snapshot.negative_percentage, snapshot.neutral_percentage],
                   color=['#4caf50', '#f44336', '#9e9e9e'])
            ax.set_ylabel('Percentage')
            ax.set_title('Sentiment Breakdown')
            sentiment_chart = fig_to_base64(fig)

            # Sentiment Over Time
            fig, ax = plt.subplots()
            dates = json.loads(snapshot.review_dates)
            pos = json.loads(snapshot.positive_scores)
            neg = json.loads(snapshot.negative_scores)
            neu = json.loads(snapshot.neutral_scores)

            min_len = min(len(dates), len(pos), len(neg), len(neu))
            dates, pos, neg, neu = dates[:min_len], pos[:min_len], neg[:min_len], neu[:min_len]

            ax.plot(dates, pos, label='Positive', color='#4caf50')
            ax.plot(dates, neg, label='Negative', color='#f44336')
            ax.plot(dates, neu, label='Neutral', color='#9e9e9e')
            ax.set_title('Sentiment Over Time')
            ax.set_ylabel('Score')
            ax.legend()
            fig.autofmt_xdate()
            trend_chart = fig_to_base64(fig)

            # Sentiment by Country
            fig, ax = plt.subplots()
            country_data = json.loads(snapshot.country_sentiment)
            countries = list(country_data.keys())
            pos_data = [country_data[c]['positive'] for c in countries]
            neg_data = [country_data[c]['negative'] for c in countries]

            x = range(len(countries))
            ax.bar(x, pos_data, label='Positive', color='#4caf50')
            ax.bar(x, neg_data, bottom=pos_data, label='Negative', color='#f44336')
            ax.set_xticks(x)
            ax.set_xticklabels(countries, rotation=45, ha='right')
            ax.set_title('Sentiment by Country')
            ax.legend()
            country_chart = fig_to_base64(fig)

        except Exception as e:
            print(f"[ERROR] Chart generation failed: {e}")

    return render_template('dashboard.html',
                           user=current_user,
                           sentiment_chart=sentiment_chart,
                           trend_chart=trend_chart,
                           country_chart=country_chart)


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_bytes = buf.read()
    encoded = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:image/png;base64,{encoded}"


# ----- My Profile -----
@main.route('/profile')
@login_required
def profile():
    return render_template('profile.html', user=current_user)


# ----- Debug: List All Users -----
@main.route('/debug-users')
def debug_users():
    if not current_app.config.get("DEBUG"):
        return jsonify({"error": "Access denied. DEBUG mode only."}), 403

    users = User.query.all()
    return jsonify([
        {"id": user.id, "email": user.email, "created_at": str(user.created_at)}
        for user in users
    ])
