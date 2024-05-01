import pandas as pd
import numpy as np

# Generate sample data
np.random.seed(42)
n_samples = 10000

degrees = ['Bachelor', 'Master', 'PhD','B.Tech']
specializations = ['Computer Science', 'Electrical Engineering', 'Business Administration', 'Data Analyst']
employed = [0, 1]  # 0 for not employed, 1 for employed
locations = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'San Francisco','California']
current_salaries = np.random.randint(300000, 1200000, size=n_samples)

data = {
    'latest_degree': np.random.choice(degrees, size=n_samples),
    'specialization': np.random.choice(specializations, size=n_samples),
    'currently_employed': np.random.choice(employed, size=n_samples),
    'current_salary': current_salaries,
    'location': np.random.choice(locations, size=n_samples),
    'recommended_for_job': np.random.randint(0, 2, size=n_samples)  # 0 for not recommended, 1 for recommended
}

# Create DataFrame
df = pd.DataFrame(data)

# Save the dataset to a CSV file
df.to_csv('job_applications1.csv', index=False)
