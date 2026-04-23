import pandas as pd
import numpy as np

np.random.seed(42)

n = 100   

data = pd.DataFrame({
    "age": np.random.randint(22, 60, n),
    "gender": np.random.choice(["Male", "Female"], n),
    "years_experience": np.random.randint(0, 20, n),
    "education_level": np.random.choice(["Bachelors", "Masters", "PhD"], n),
    "department": np.random.choice(["IT", "HR", "Finance", "Sales"], n),
    "job_role": np.random.choice(["Manager", "Executive", "Analyst"], n),
    "performance_rating": np.random.randint(1, 6, n),
    "projects_completed": np.random.randint(1, 20, n),
    "certifications": np.random.randint(0, 10, n),
    "weekly_hours": np.random.randint(30, 60, n),
    "overtime_hours": np.random.randint(0, 20, n),
    "team_size": np.random.randint(1, 15, n),
    "leadership_score": np.random.uniform(1, 10, n),
    "skill_match_score": np.random.uniform(1, 10, n),
    "promotion_count": np.random.randint(0, 5, n),
    "employment_type": np.random.choice(["Full-time", "Part-time"], n),
    "location": np.random.choice(["Urban", "Rural"], n),
    "company_size": np.random.choice(["Small", "Medium", "Large"], n),
    "job_satisfaction": np.random.randint(1, 6, n)
})

data["salary"] = (
    data["years_experience"] * 4000 +
    data["performance_rating"] * 3000 +
    data["leadership_score"] * 2000 +
    np.random.randint(20000, 40000, n)
)

data.to_csv("employee_salary_dataset.csv", index=False)

print(data.head())
print(data.shape)