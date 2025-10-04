#import "common/lib/contentBox.typ": contentBox
#import "common/lib/slideLayouts.typ": theme, mainTitle, layoutA, layoutATwoCols, layoutATwoColsWithTitle, layoutBTwoCols, layoutC, layoutCTwoColsWithTitle, layoutDTwoCols

#mainTitle(
  title: "CSDS-352 Vault",
  subtitle: "Santiago Osorio Parra <santiago.osorio@jala.university>",
  content: [
    #v(160pt)
    #contentBox(
      fill: luma(150, 30%),
      text(
        fill: white,
        size: 20pt,
        [
          This is a companion document that contains notes and references for the *Machine Learning* course.
        ],
      ),
    )
  ],
)

#mainTitle(
  title: "Capstone Project",
  subtitle: "Music Samples Search System",
)

#layoutC(
  title: "Project Overview",
  content: [
    I chose to build a *Music Samples Search System* using audio embeddings and vector databases to explore how machine learning can understand audio similarity and enable content-based music retrieval.
    
    == Goals & Technology
    - Extract 45-dimensional audio embeddings from MP3 files
    - Store and search using ChromaDB vector database
    - Build REST API with FastAPI for similarity search
    - Evaluate using classification and anomaly detection
    
    *Tech Stack*: Python, FastAPI, ChromaDB 0.4.15, Librosa, Docker
    
    *Dataset*: 267 audio samples across 6 electronic music genres (House: 50, Drum & Bass: 50, Ambient: 47, Techno: 44, Trance: 42, Dubstep: 34)
  ],
)

#layoutC(
  title: "Problem Domain & Applications",
  content: [
    == The Music Production Challenge
    
    Music producers and DJs work with thousands of audio samples daily. Finding the right sound manually is time-consuming and inefficient. Current solutions rely on text tags and metadata, which are often incomplete, inconsistent, or subjective.
    
    *Key Problems*:
    - Manual search through large sample libraries takes hours
    - Text-based search misses acoustic similarities
    - Similar sounds have different tags across libraries
    - No way to find "sounds like this" samples
    
    == Real-World Applications
    
    *Music Production*: Producers can upload a reference sound and instantly find similar samples from their library, speeding up creative workflow.
    
    *Sample Libraries*: Companies like Splice, Loopmasters can offer "find similar" features, improving user experience and discovery.
    
    *DJ Sets*: DJs can quickly find tracks with similar energy, tempo, or mood for seamless mixing.
    
    *Copyright Detection*: Identify potential sample usage and derivative works by acoustic similarity rather than metadata.
    
    *Music Education*: Students can explore genre characteristics by finding similar examples and understanding audio patterns.
  ],
)

#layoutCTwoColsWithTitle(
  title: "System Architecture & API",
  contentLeft: [
    == Components
    
    *Audio Feature Extraction*
    - Processes MP3 files
    - Generates 45-dim embeddings
    - MFCCs, spectral features, tempo
    
    *ChromaDB Vector Database*
    - 267 sample embeddings
    - Cosine distance search
    - Metadata storage
    
    *FastAPI REST Endpoints*
    - `/stats`: Database info
    - `/search`: k-NN similarity
    - `/upload`: Add samples
  ],
  contentRight: [
    == Infrastructure
    
    Two Docker containers:
    - `music-backend` (port 8001)
    - `music-chromadb` (port 8000)
    
    == API Usage
    
    ```bash
    # Get statistics
    curl http://localhost:8001/stats
    
    # Search similar
    curl -X POST \
      -F "file=@sample.mp3" \
      "http://localhost:8001/search?n_results=5"
    ```
  ],
)

#layoutC(
  title: "Search Quality Results",
  content: [
    Tested with samples from three different genres to evaluate similarity search performance:
    
    #table(
      columns: (auto, 1fr, auto, auto),
      stroke: 0.5pt,
      [*Query*], [*Genre*], [*Duration*], [*Key Finding*],
      [techno_100385.mp3], [Techno], [0.87s], [Perfect self-match (d=0), some house confusion],
      [ambient_105409.mp3], [Ambient], [3.96s], [Top 3 all ambient, larger distance variance],
      [house_113625.mp3], [House], [0.48s], [Excellent match, techno confusion (d=0.11)],
    )
    
    *Observations*: System achieves perfect self-matching (distance ≈ 0). Cross-genre confusion reflects real acoustic similarities between house/techno and ambient/drum & bass. Distance metrics vary with sample duration.
  ],
)

#layoutCTwoColsWithTitle(
  title: "Classification & Anomaly Detection",
  contentLeft: [
    == Genre Classification
    
    *Setup*: 213 train / 54 test samples, k-NN classifier
    
    #contentBox(
      fill: luma(150, 30%),
      text(fill: white, [
        *Overall Accuracy: 40.74%*
      ])
    )
    
    #table(
      columns: (1fr, auto),
      stroke: 0.5pt,
      [*Genre*], [*F1-Score*],
      [Ambient], [66.67%],
      [House], [60.00%],
      [Techno], [37.50%],
      [Trance], [30.77%],
      [Drum & Bass], [23.08%],
      [Dubstep], [16.67%],
    )
    
    Low accuracy suggests overlapping feature spaces in electronic music.
  ],
  contentRight: [
    == Anomaly Detection
    
    *Algorithm*: Isolation Forest (10% contamination = 27 outliers)
    
    Top anomalies dominated by Ambient and Drum & Bass genres, suggesting higher intra-genre variability.
    
    *Top 5 Anomalies*:
    - drum_and_bass_114994.mp3 (-0.746)
    - ambient_269592.mp3 (-0.699)
    - drum_and_bass_114986.mp3 (-0.683)
    - ambient_802900.mp3 (-0.680)
    - ambient_28838.mp3 (-0.673)
    
    Useful for quality control and discovering unique samples.
  ],
)

#layoutC(
  title: "Challenges & Future Work",
  content: [
    == Current Limitations
    
    *Classification*: 40.74% accuracy due to genre overlap and short sample durations (0.48s-3.96s)
    
    *Embeddings*: 45 dimensions may not capture full audio complexity
    
    *Dataset*: Small size (267 samples) with unbalanced distribution (34-50 per genre)
    
    == Planned Improvements
    
    *Short-term*: Increase embedding dimensions to 128+, expand dataset to 1000+ samples per genre, experiment with pre-trained models (VGGish, OpenL3)
    
    *Long-term*: Multi-modal search (audio + text), web-based UI with visualization, production deployment with scalability
  ],
)

#layoutC(
  title: "Conclusions & Lessons Learned",
  content: [
    == Key Achievements
    
    Successfully built and deployed a working music similarity search system with REST API. Evaluated performance using multiple metrics (search quality, classification, anomaly detection).
    
    == Main Findings
    
    System achieves perfect self-matching for search. Electronic music genres share acoustic characteristics making classification challenging (40.74% accuracy). Ambient and House easiest to classify (60-67% F1), Dubstep and Drum & Bass most difficult (17-23% F1).
    
    == Lessons Learned
    
    *ML Insights*: Audio embeddings quality directly impacts performance. Pre-trained models could significantly improve results. Dataset quality matters more than quantity.
    
    *Engineering*: ChromaDB enables fast similarity search. Docker simplifies deployment. Clear API design essential for production systems.
  ],
)