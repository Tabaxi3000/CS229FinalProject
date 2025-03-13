package collector;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import core.Card;

public class RLDataCollector {
    private final CopyOnWriteArrayList<GameState> gameStates;
    private final Gson gson;
    private int currentFileGameId;  // Track game IDs within each file
    
    public RLDataCollector() {
        this.gameStates = new CopyOnWriteArrayList<>();
        this.gson = new GsonBuilder().setPrettyPrinting().create();
        this.currentFileGameId = 0;
    }
    
    public void recordGameState(int gameId, int turnNumber, int currentPlayer,
                              ArrayList<Card> playerHand, ArrayList<Card> knownOpponentCards,
                              Card faceUpCard, ArrayList<Card> discardPile,
                              String action, double reward, boolean isTerminal,
                              int deadwoodPoints) {
        // Use the original game ID directly instead of mapping to a file-local ID
        // This ensures unique game IDs across all files
        
        GameState state = new GameState(gameId, turnNumber, currentPlayer,
            playerHand, knownOpponentCards, faceUpCard, discardPile,
            action, reward, isTerminal, deadwoodPoints);
        gameStates.add(state);
    }
    
    public void saveToFile(String filename) {
        try (FileWriter writer = new FileWriter(filename)) {
            gson.toJson(gameStates, writer);
            System.out.println("Saved " + gameStates.size() + " game states to " + filename);
            System.out.println("Number of unique games in file: " + (currentFileGameId + 1));
        } catch (IOException e) {
            System.err.println("Error saving data to " + filename + ": " + e.getMessage());
        }
    }
    
    public void clear() {
        gameStates.clear();
        currentFileGameId = 0;  // Reset game ID counter for the new file
    }
    
    public void incrementGameId() {
        currentFileGameId++;
    }
    
    private static class GameState {
        private final int gameId;
        private final int turnNumber;
        private final int currentPlayer;
        private final ArrayList<Integer> playerHand;  // Card indices
        private final ArrayList<Integer> knownOpponentCards;
        private final int faceUpCard;
        private final ArrayList<Integer> discardPile;
        private final String action;
        private final double reward;
        private final boolean isTerminal;
        private final int deadwoodPoints;
        
        public GameState(int gameId, int turnNumber, int currentPlayer,
                        ArrayList<Card> playerHand, ArrayList<Card> knownOpponentCards,
                        Card faceUpCard, ArrayList<Card> discardPile,
                        String action, double reward, boolean isTerminal,
                        int deadwoodPoints) {
            this.gameId = gameId;
            this.turnNumber = turnNumber;
            this.currentPlayer = currentPlayer;
            this.playerHand = new ArrayList<>();
            this.knownOpponentCards = new ArrayList<>();
            this.discardPile = new ArrayList<>();
            
            // Convert Card objects to indices
            for (Card card : playerHand) {
                this.playerHand.add(card.getId());
            }
            for (Card card : knownOpponentCards) {
                this.knownOpponentCards.add(card.getId());
            }
            this.faceUpCard = faceUpCard != null ? faceUpCard.getId() : -1;
            for (Card card : discardPile) {
                this.discardPile.add(card.getId());
            }
            
            this.action = action;
            this.reward = reward;
            this.isTerminal = isTerminal;
            this.deadwoodPoints = deadwoodPoints;
        }
    }
} 