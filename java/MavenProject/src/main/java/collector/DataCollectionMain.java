package collector;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import core.GinRummyGame;
import player.RLDataCollectionPlayer;
import player.SimplePlayer;

public class DataCollectionMain {
    private static final int REMAINING_GAMES = 13082;  // We need 13,082 more games to reach 100,000
    private static final int GAMES_PER_FILE = 10000;
    private static final int NUM_THREADS = Runtime.getRuntime().availableProcessors();
    private static final String OUTPUT_FILE_PATTERN = "training_data_remaining_%d.json";  // New file pattern to avoid overwriting
    private static final Object fileLock = new Object();  // Lock for file operations
    private static final Object collectorLock = new Object();  // Lock for collector operations
    
    // Start game IDs from a value higher than any existing game ID
    private static final int STARTING_GAME_ID = 60000;  // Higher than the highest existing ID (57196)
    
    public static void main(String[] args) {
        System.out.println("Starting remaining data collection...");
        System.out.println("Number of games to generate: " + REMAINING_GAMES);
        System.out.println("Using " + NUM_THREADS + " threads");
        System.out.println("Games per thread: " + (REMAINING_GAMES / NUM_THREADS));
        System.out.println("Starting game ID: " + STARTING_GAME_ID);
        
        ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
        AtomicInteger gamesPlayed = new AtomicInteger(0);
        AtomicInteger dataPlayerWins = new AtomicInteger(0);
        AtomicInteger currentFileGames = new AtomicInteger(0);
        AtomicInteger fileCounter = new AtomicInteger(0);
        AtomicInteger nextGameId = new AtomicInteger(STARTING_GAME_ID);
        
        // Create a single shared collector
        RLDataCollector sharedCollector = new RLDataCollector();
        
        // Split games among threads
        int gamesPerThread = REMAINING_GAMES / NUM_THREADS;
        for (int t = 0; t < NUM_THREADS; t++) {
            final int threadId = t;
            executor.submit(() -> {
                try {
                    System.out.printf("Thread %d starting, will play games %d to %d\n", 
                        threadId, threadId * gamesPerThread, 
                        (threadId == NUM_THREADS - 1) ? REMAINING_GAMES - 1 : (threadId + 1) * gamesPerThread - 1);
                    
                    Random random = new Random();
                    int startGame = threadId * gamesPerThread;
                    int endGame = (threadId == NUM_THREADS - 1) ? REMAINING_GAMES : (threadId + 1) * gamesPerThread;
                    
                    for (int i = startGame; i < endGame; i++) {
                        // Get a unique game ID for this game
                        int gameId = nextGameId.getAndIncrement();
                        
                        // Create new players and game for each iteration
                        RLDataCollectionPlayer dataPlayer = new RLDataCollectionPlayer(sharedCollector, gameId);
                        SimplePlayer simplePlayer = new SimplePlayer();
                        GinRummyGame game = new GinRummyGame(dataPlayer, simplePlayer);
                        
                        System.out.printf("Thread %d playing game %d (ID: %d)\n", threadId, i, gameId);
                        int winner = game.play();
                        int currentPlayed = gamesPlayed.incrementAndGet();
                        if (winner == 0) dataPlayerWins.incrementAndGet();
                        
                        // Save data when we've collected GAMES_PER_FILE games
                        synchronized(collectorLock) {
                            sharedCollector.incrementGameId();  // Increment game ID after each game
                            int gamesInCurrentFile = currentFileGames.incrementAndGet();
                            if (gamesInCurrentFile >= GAMES_PER_FILE) {
                                String filename = String.format(OUTPUT_FILE_PATTERN, fileCounter.incrementAndGet());
                                sharedCollector.saveToFile(filename);
                                sharedCollector.clear();  // This also resets the game ID counter
                                currentFileGames.set(0);
                                
                                // Print progress
                                System.out.printf("Saved file %s - Total games: %d/%d (%.1f%%), Win rate: %.2f%%\n", 
                                    filename, currentPlayed, REMAINING_GAMES,
                                    (double)currentPlayed/REMAINING_GAMES * 100,
                                    (double)dataPlayerWins.get()/currentPlayed * 100);
                            }
                        }
                    }
                    
                    System.out.printf("Thread %d completed its games\n", threadId);
                } catch (Exception e) {
                    System.err.printf("Thread %d encountered error: %s\n", threadId, e.toString());
                    e.printStackTrace();
                }
            });
        }
        
        // Wait for all threads to complete
        executor.shutdown();
        try {
            executor.awaitTermination(24, TimeUnit.HOURS);
            
            // Save any remaining games
            if (currentFileGames.get() > 0) {
                String filename = String.format(OUTPUT_FILE_PATTERN, fileCounter.incrementAndGet());
                sharedCollector.saveToFile(filename);
            }
        } catch (InterruptedException e) {
            System.err.println("Data collection interrupted!");
            return;
        }
        
        System.out.println("\nData collection complete!");
        System.out.printf("Final stats - Games played: %d, Win rate: %.2f%%\n",
            gamesPlayed.get(),
            (double)dataPlayerWins.get()/gamesPlayed.get() * 100);
        System.out.printf("Data saved in %d files using pattern: %s\n", 
            fileCounter.get(), OUTPUT_FILE_PATTERN);
        System.out.println("Total games generated (including previous runs): 100,000");
        System.out.printf("Game IDs range: %d to %d\n", STARTING_GAME_ID, nextGameId.get() - 1);
    }
} 