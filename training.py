import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import torch.optim as optim
from sklearn.model_selection import train_test_split



# Dataset class
class MovieLensDataset(Dataset):
    def __init__(self, df):
        self.user_indices = torch.tensor(df['user_idx'].values, dtype=torch.long)
        self.movie_indices = torch.tensor(df['movie_idx'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_indices[idx], self.movie_indices[idx], self.ratings[idx]






# Create directory if not exists
os.makedirs('models', exist_ok=True)

# Load data
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')

# Compute mean and std
mean_rating = ratings_df['rating'].mean()
std_rating = ratings_df['rating'].std()

# Normalize ratings
ratings_df['rating'] = (ratings_df['rating'] - mean_rating) / std_rating

# Split ratings into train and validation sets (e.g., 80% train, 20% val)
train_df, val_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

# Map movieId and userId to indices (starting from 0)
unique_movie_ids = ratings_df['movieId'].unique()
unique_user_ids = ratings_df['userId'].unique()

movie2idx = {mid: idx for idx, mid in enumerate(unique_movie_ids)}
user2idx = {uid: idx for idx, uid in enumerate(unique_user_ids)}

ratings_df['movie_idx'] = ratings_df['movieId'].map(movie2idx)
ratings_df['user_idx'] = ratings_df['userId'].map(user2idx)

num_movies = len(movie2idx)
num_users = len(user2idx)

print(f"Number of movies: {num_movies}, Number of users: {num_users}")


train_df['user_idx'] = train_df['userId'].map(user2idx)
train_df['movie_idx'] = train_df['movieId'].map(movie2idx)
val_df['user_idx'] = val_df['userId'].map(user2idx)
val_df['movie_idx'] = val_df['movieId'].map(movie2idx)

# Create datasets and dataloaders
train_dataset = MovieLensDataset(train_df)
val_dataset = MovieLensDataset(val_df)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)





# Define model
class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.movie_emb = nn.Embedding(num_movies, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)
        
    def forward(self, user_indices, movie_indices):
        user_vecs = self.user_emb(user_indices)
        movie_vecs = self.movie_emb(movie_indices)
        user_b = self.user_bias(user_indices).squeeze()
        movie_b = self.movie_bias(movie_indices).squeeze()
        
        dot = (user_vecs * movie_vecs).sum(dim=1) + user_b + movie_b
        return dot







# Instantiate model
embedding_dim = 32
learning_rate = 0.005
model = RecommenderNet(num_users, num_movies, embedding_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


def init_weights(m):
    if type(m) == nn.Embedding:
        nn.init.xavier_normal_(m.weight)
model.apply(init_weights)



# Training
epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for user_idx, movie_idx, rating in train_loader:
        optimizer.zero_grad()
        preds = model(user_idx, movie_idx)
        loss = criterion(preds, rating)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * user_idx.size(0)
    avg_loss = total_loss / len(train_dataset)

    scheduler.step()
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}")


# Validation
model.eval()
val_loss = 0    
with torch.no_grad():
    for user_idx, movie_idx, rating in val_loader:
        preds = model(user_idx, movie_idx)
        loss = criterion(preds, rating)
        val_loss += loss.item() * user_idx.size(0)
avg_val_loss = val_loss / len(val_dataset)
print(f"Val Loss: {avg_val_loss:.4f}")




# Save normalization stats along with mappings
mappings = {
    'movie2idx': movie2idx,
    'user2idx': user2idx,
    'movieId2title': dict(zip(movies_df['movieId'], movies_df['title'])),
    'mean_rating': mean_rating,
    'std_rating': std_rating
}

torch.save(model.state_dict(), 'models/recommender.pth')

with open('models/mappings.pkl', 'wb') as f:
    pickle.dump(mappings, f)

print("Training complete and model saved.")