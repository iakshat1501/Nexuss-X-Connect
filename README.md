# 🔗 Nexuss X Connect

Nexuss X Connect is an intelligent alumni-student interaction portal that uses a machine learning-based recommendation engine to connect students with the most relevant alumni. It enables mentorship, networking, and career guidance by matching users based on skill sets, interests, and professional background.

---

## 🚀 Features

- 🎯 **Recommendation System**: Smart ML model (`recommendation_model.pkl`) to suggest the most relevant alumni-student matches.
- 🌐 **User Interface**: A clean and minimal front-end using `index.html` for easy access and interaction.
- 📄 **Data-Driven**: Uses real-world data (`updated_linkedin.csv`) for modeling alumni profiles.
- 🧠 **Built with Python**: Backend powered by Flask and machine learning libraries.

---

## 📁 Project Structure
Nexuss-X-Connect/
├── app.py # Flask backend server
├── index.html # Web UI
├── updated_linkedin.csv # Dataset of alumni profiles
├── recommendation_model.pkl # Trained ML model for user recommendations


---

## 💡 How It Works

1. **User Login**: Students or alumni log into the portal.
2. **Profile Input**: The system captures key details like skills, industry, and experience.
3. **Recommendation Engine**: ML model finds best-matched connections from alumni database.
4. **Connection Suggestions**: Displays recommended alumni with links to connect.

---

## 🛠 Tech Stack

- **Frontend**: HTML, CSS
- **Backend**: Python (Flask)
- **Machine Learning**: Scikit-learn, Pandas
- **Deployment Ready**: Can be hosted on Vercel, Heroku, or AWS EC2

---

## 🧪 Getting Started

### 🔧 Prerequisites

- Python 3.7+
- Flask: `pip install flask`
- Scikit-learn, Pandas: `pip install -r requirements.txt` *(Create one if not present)*

📊 Dataset Preview
The dataset updated_linkedin.csv includes:

Name

Skills

Experience

Industry

LinkedIn URL

Used to train and test the recommendation engine.

✨ Future Enhancements
Add user authentication & dashboards

Real-time chat or appointment scheduling

Enhanced NLP-based profile matching

Deployment to cloud platforms (AWS/GCP)

👨‍💻 Author
Akshat Garg
Smart India Hackathon 2022 - Finalist

### ▶️ Run Locally

```bash
git clone https://github.com/your-username/Nexuss-X-Connect.git
cd Nexuss-X-Connect
python app.py


