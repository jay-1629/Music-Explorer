import customtkinter as ctk
import pandas as pd
import numpy as np
import webbrowser
import urllib.parse
import os
import random
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from thefuzz import process

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class MusicApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Music Explorer")
        self.geometry("1100x850")

        self.load_data()
        self.cooldown_active = False

        #sidebar
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        
        ctk.CTkLabel(self.sidebar, text="Recent Searches", font=("Arial", 16, "bold")).pack(pady=(20, 5))
        self.history_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.history_frame.pack(fill="x", padx=10)
        self.history_list = []
        
        ctk.CTkLabel(self.sidebar, text="Filter Genres", font=("Arial", 16, "bold")).pack(pady=(20, 5))
        self.genre_scroll = ctk.CTkScrollableFrame(self.sidebar, height=400, fg_color="#1e1e1e")
        self.genre_scroll.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.genre_vars = {}
        self.setup_genre_filters()
        self.load_history()

        #main area
        self.main_container = ctk.CTkFrame(self)
        self.main_container.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.entry = ctk.CTkEntry(self.main_container, placeholder_text="Enter song name...", width=400, height=40)
        self.entry.pack(pady=20)
        self.entry.bind("<Return>", lambda event: self.start_search())

        self.btn_row = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.btn_row.pack(fill="x", padx=100)
        
        self.btn = ctk.CTkButton(self.btn_row, text="Search", command=self.start_search, fg_color="#1DB954")
        self.btn.pack(side="left", padx=10, expand=True)
        
        self.random_btn = ctk.CTkButton(self.btn_row, text="🎲 Random", command=self.search_random, fg_color="#535353")
        self.random_btn.pack(side="right", padx=10, expand=True)

        #artist dropdown (hidden by default)
        self.artist_dropdown = ctk.CTkOptionMenu(self.main_container, width=400, command=self.on_artist_selected)
        #not packed

        self.results_frame = ctk.CTkScrollableFrame(self.main_container, width=700, height=500)
        self.results_frame.pack(pady=20, padx=20, fill="both", expand=True)

    def load_data(self):
        self.df = pd.read_csv("dataset.csv")
        self.df = self.df.drop_duplicates(subset=['track_name', 'artists']).reset_index(drop=True)
        features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
                    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
        genre_dummies = pd.get_dummies(self.df['track_genre'], prefix='genre')
        scaler = StandardScaler()
        scaled_audio = scaler.fit_transform(self.df[features])
        self.X = np.hstack([scaled_audio, genre_dummies.values * 3.0])
        self.model = NearestNeighbors(n_neighbors=100, metric='cosine')
        self.model.fit(self.X)

    def setup_genre_filters(self):
        unique_genres = sorted(self.df['track_genre'].unique().tolist())
        for g in unique_genres:
            var = ctk.BooleanVar(value=True)
            cb = ctk.CTkCheckBox(self.genre_scroll, text=g, variable=var, font=("Arial", 11), checkbox_height=18, checkbox_width=18)
            cb.pack(anchor="w", padx=10, pady=2) 
            self.genre_vars[g] = var

    def start_search(self):
        query = self.entry.get().strip()
        if not query: return
        self.btn.configure(text="Searching...", state="disabled")
        self.update_idletasks()
        
        try:
            self.clear_results()
            self.artist_dropdown.pack_forget() #hide the dropdown on new search

            all_names = self.df['track_name'].unique().tolist()
            match_result = process.extractOne(query, all_names[:40000])
            
            if match_result and match_result[1] >= 60:
                self.current_song_name = match_result[0]
                #finding all rows matching this exact name
                self.matches = self.df[self.df['track_name'] == self.current_song_name].sort_values('popularity', ascending=False)

                if len(self.matches) > 1:
                    #logic for when multiple artists are found
                    artist_options = [f"{row.artists} (Pop: {row.popularity})" for row in self.matches.itertuples()]
                    self.artist_dropdown.configure(values=artist_options)
                    self.artist_dropdown.set(artist_options[0])
                    self.artist_dropdown.pack(pady=10, after=self.btn_row)
                    self.add_text_to_results("Multiple artists found. Please select one from the dropdown.")
                else:
                    self.show_recommendations(self.matches.iloc[0])
            else:
                self.add_text_to_results("❌ No match found.")
        finally:
            self.btn.configure(text="Search", state="normal")

    def on_artist_selected(self, selection):
        self.clear_results()
        #extract index from the selection string list
        idx_in_matches = self.artist_dropdown._values.index(selection)
        selected_row = self.matches.iloc[idx_in_matches]
        self.show_recommendations(selected_row)

    def search_random(self):
        if self.cooldown_active: return
        self.start_cooldown(3)
        self.artist_dropdown.pack_forget()
        top_songs = self.df.nlargest(2000, 'popularity')
        random_song = top_songs.iloc[random.randint(0, 1999)]
        self.entry.delete(0, 'end')
        self.entry.insert(0, random_song['track_name'])
        self.show_recommendations(random_song)

    def show_recommendations(self, selected_row):
        self.save_history(f"{selected_row['track_name']} - {selected_row['artists']}")
        query_vec = self.X[selected_row.name].reshape(1, -1)
        distances, indices = self.model.kneighbors(query_vec)
        self.clear_results()
        self.after(50, lambda: self.process_recommendations(indices, selected_row))

    def process_recommendations(self, indices, selected_row):
        active_genres = [g for g, var in self.genre_vars.items() if var.get()]
        count = 0
        for i in range(1, len(indices[0])):
            if count >= 10: break
            idx = indices[0][i]
            rec_row = self.df.iloc[idx]
            if rec_row['track_genre'] in active_genres:
                if rec_row['track_name'].lower() != selected_row['track_name'].lower():
                    self.create_result_card(rec_row, selected_row)
                    count += 1
        if count == 0:
            self.add_text_to_results("No matches found within allowed genres.")

    def create_result_card(self, row, original):
        diffs = {f: abs(original[f] - row[f]) for f in ['energy', 'danceability', 'valence']}
        best_f = min(diffs, key=diffs.get)
        explanation = f"Matched via {best_f.capitalize()} | Genre: {row['track_genre']}"

        #the card
        card = ctk.CTkFrame(self.results_frame, fg_color="#2b2b2b")
        card.pack(fill="x", padx=10, pady=5)
        
        #left side of card - TEXT INFO
        text_frame = ctk.CTkFrame(card, fg_color="transparent")
        text_frame.pack(side="left", padx=15, pady=10, fill="both", expand=True)

        ctk.CTkLabel(text_frame, text=row['track_name'], font=("Arial", 13, "bold"), anchor="w").pack(fill="x")
        ctk.CTkLabel(text_frame, text=row['artists'], font=("Arial", 11), anchor="w").pack(fill="x")
        ctk.CTkLabel(text_frame, text=explanation, font=("Arial", 10), text_color="#1DB954", anchor="w").pack(fill="x")

        #right side of card - PLAY BUTTON
        search_term = urllib.parse.quote(f"{row['track_name']} {row['artists']}")
        play_btn = ctk.CTkButton(card, text="▶ Play", width=70, height=32, fg_color="#333333", 
                                command=lambda url=search_term: webbrowser.open(f"https://www.youtube.com/results?search_query={url}"))
        
        #pady to make sure it stays centered, even if text grows
        play_btn.pack(side="right", padx=15, pady=10)

    def start_cooldown(self, seconds):
        self.cooldown_active = True
        self.random_btn.configure(state="disabled")
        self.update_cooldown_text(seconds)

    def update_cooldown_text(self, seconds):
        if seconds > 0:
            self.random_btn.configure(text=f"Wait ({seconds}s)")
            self.after(1000, lambda: self.update_cooldown_text(seconds - 1))
        else:
            self.cooldown_active = False
            self.random_btn.configure(state="normal", text="🎲 Random")

    def save_history(self, item):
        if item in self.history_list: self.history_list.remove(item)
        self.history_list.insert(0, item)
        self.history_list = self.history_list[:5]
        try:
            with open("history.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(self.history_list))
        except: pass
        self.refresh_history_ui()

    def load_history(self):
        if os.path.exists("history.txt"):
            try:
                with open("history.txt", "r", encoding="utf-8") as f:
                    self.history_list = [line.strip() for line in f.readlines() if line.strip()]
                self.refresh_history_ui()
            except: pass

    def refresh_history_ui(self):
        for w in self.history_frame.winfo_children(): w.destroy()
        for item in self.history_list:
            btn = ctk.CTkButton(self.history_frame, text=item, font=("Arial", 10), fg_color="transparent", anchor="w", 
                                command=lambda i=item: self.search_from_history(i))
            btn.pack(fill="x")

    def search_from_history(self, item):
        song_name = item.split(" - ")[0]
        self.entry.delete(0, 'end')
        self.entry.insert(0, song_name)
        self.start_search()

    def clear_results(self):
        for w in self.results_frame.winfo_children(): w.destroy()

    def add_text_to_results(self, text):
        ctk.CTkLabel(self.results_frame, text=text).pack(pady=40)

if __name__ == "__main__":
    app = MusicApp()
    app.mainloop()