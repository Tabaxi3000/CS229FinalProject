package player;

import java.util.ArrayList;
import java.util.Random;
import java.util.Stack;

import collector.RLDataCollector;
import core.Card;
import core.GinRummyPlayer;
import core.GinRummyUtil;

public class RLDataCollectionPlayer implements GinRummyPlayer {
    private int playerNum;
    private int startingPlayerNum;
    private ArrayList<Card> cards = new ArrayList<>();
    private ArrayList<Card> knownOpponentCards = new ArrayList<>();
    private ArrayList<Card> discardPile;
    private Card faceUpCard;
    private Card drawnCard;
    private Random random = new Random();
    private RLDataCollector collector;
    private final int gameId;
    private int turnNumber;
    private boolean gameEnded;
    private boolean opponentKnocked = false;
    private ArrayList<Long> drawDiscardBitstrings = new ArrayList<Long>();
    
    public RLDataCollectionPlayer(RLDataCollector collector, int gameId) {
        this.collector = collector;
        this.gameId = gameId;
    }
    
    @Override
    public void startGame(int playerNum, int startingPlayerNum, Card[] cards) {
        this.playerNum = playerNum;
        this.startingPlayerNum = startingPlayerNum;
        this.cards.clear();
        for (Card card : cards)
            this.cards.add(card);
        this.knownOpponentCards.clear();
        this.discardPile = new ArrayList<>();
        this.turnNumber = 0;
        this.gameEnded = false;
        this.opponentKnocked = false;
        this.drawDiscardBitstrings.clear();
    }
    
    @Override
    public boolean willDrawFaceUpCard(Card card) {
        this.faceUpCard = card;
        
        // Check if card would improve our hand
        ArrayList<Card> newCards = new ArrayList<>(cards);
        newCards.add(card);
        
        // Get current deadwood
        ArrayList<ArrayList<ArrayList<Card>>> currentMeldSets = GinRummyUtil.cardsToBestMeldSets(cards);
        int currentDeadwood = currentMeldSets.isEmpty() ? 
            GinRummyUtil.getDeadwoodPoints(cards) : 
            GinRummyUtil.getDeadwoodPoints(currentMeldSets.get(0), cards);
            
        // Get potential deadwood with new card
        ArrayList<ArrayList<ArrayList<Card>>> newMeldSets = GinRummyUtil.cardsToBestMeldSets(newCards);
        int newDeadwood = newMeldSets.isEmpty() ? 
            GinRummyUtil.getDeadwoodPoints(newCards) : 
            GinRummyUtil.getDeadwoodPoints(newMeldSets.get(0), newCards);
            
        boolean improvesMeld = false;
        for (ArrayList<Card> meld : GinRummyUtil.cardsToAllMelds(newCards)) {
            if (meld.contains(card)) {
                improvesMeld = true;
                break;
            }
        }
        
        // Draw if it reduces deadwood or forms a meld (90% chance if significant improvement)
        boolean willDraw = false;
        if (improvesMeld || newDeadwood < currentDeadwood - 2) {
            willDraw = random.nextDouble() < 0.9;
        } else if (newDeadwood <= currentDeadwood) {
            willDraw = random.nextDouble() < 0.4;
        } else {
            willDraw = random.nextDouble() < 0.1;
        }
        
        // Record state before decision
        recordGameState("draw_faceup_" + willDraw, 0.0, false);
        
        return willDraw;
    }
    
    @Override
    public void reportDiscard(int playerNum, Card discardedCard) {
        if (playerNum != this.playerNum) {
            knownOpponentCards.remove(discardedCard);
        }
        discardPile.add(0, discardedCard);
        if (playerNum == this.playerNum)
            cards.remove(discardedCard);
    }
    
    @Override
    public void reportDraw(int playerNum, Card drawnCard) {
        if (playerNum == this.playerNum) {
            cards.add(drawnCard);
            this.drawnCard = drawnCard;
        } else if (drawnCard != null) {
            knownOpponentCards.add(drawnCard);
        }
    }
    
    @Override
    public Card getDiscard() {
        turnNumber++;
        
        // Find cards that would leave us with minimal deadwood while preserving melds
        int minDeadwood = Integer.MAX_VALUE;
        ArrayList<Card> bestDiscards = new ArrayList<>();
        
        for (Card card : cards) {
            // Cannot draw and discard face up card
            if (card == drawnCard && drawnCard == faceUpCard)
                continue;
                
            // Prevent repeat of draw and discard
            ArrayList<Card> drawDiscard = new ArrayList<>();
            drawDiscard.add(drawnCard);
            drawDiscard.add(card);
            if (drawDiscardBitstrings.contains(GinRummyUtil.cardsToBitstring(drawDiscard)))
                continue;
            
            ArrayList<Card> remainingCards = new ArrayList<>(cards);
            remainingCards.remove(card);
            ArrayList<ArrayList<ArrayList<Card>>> bestMeldSets = GinRummyUtil.cardsToBestMeldSets(remainingCards);
            
            // Calculate deadwood considering melds
            int deadwood = bestMeldSets.isEmpty() ? 
                GinRummyUtil.getDeadwoodPoints(remainingCards) : 
                GinRummyUtil.getDeadwoodPoints(bestMeldSets.get(0), remainingCards);
            
            if (deadwood <= minDeadwood) {
                if (deadwood < minDeadwood) {
                    minDeadwood = deadwood;
                    bestDiscards.clear();
                }
                bestDiscards.add(card);
            }
        }
        
        // Choose randomly from the best discards (90%) or any card (10%)
        Card discardCard;
        if (random.nextDouble() < 0.9 && !bestDiscards.isEmpty()) {
            discardCard = bestDiscards.get(random.nextInt(bestDiscards.size()));
        } else {
            discardCard = cards.get(random.nextInt(cards.size()));
        }
        
        // Prevent future repeat of draw, discard pair
        ArrayList<Card> drawDiscard = new ArrayList<>();
        drawDiscard.add(drawnCard);
        drawDiscard.add(discardCard);
        drawDiscardBitstrings.add(GinRummyUtil.cardsToBitstring(drawDiscard));
        
        // Record state after discard decision
        recordGameState("discard_" + discardCard.getId(), 0.0, false);
        
        return discardCard;
    }
    
    @Override
    public void reportFinalMelds(int playerNum, ArrayList<ArrayList<Card>> melds) {
        if (playerNum != this.playerNum) {
            opponentKnocked = true;
            for (ArrayList<Card> meld : melds) {
                knownOpponentCards.addAll(meld);
            }
        }
        
        // Only record game over state after both players have reported their melds
        if (playerNum == this.playerNum && !gameEnded) {
            gameEnded = true;
            // We'll set the reward in reportScores when we know the actual winner
            recordGameState("game_over", 0.0, true);
        }
    }
    
    @Override
    public ArrayList<ArrayList<Card>> getFinalMelds() {
        // Get best possible melds
        ArrayList<ArrayList<ArrayList<Card>>> bestMeldSets = GinRummyUtil.cardsToBestMeldSets(cards);
        
        // Check if deadwood is low enough to go out
        if (!opponentKnocked && (bestMeldSets.isEmpty() || 
            GinRummyUtil.getDeadwoodPoints(bestMeldSets.get(0), cards) > GinRummyUtil.MAX_DEADWOOD)) {
            return null;
        }
        
        return bestMeldSets.isEmpty() ? 
            new ArrayList<ArrayList<Card>>() : 
            bestMeldSets.get(random.nextInt(bestMeldSets.size()));
    }
    
    private void recordGameState(String action, double reward, boolean gameOver) {
        if (collector != null) {
            ArrayList<ArrayList<ArrayList<Card>>> bestMeldSets = GinRummyUtil.cardsToBestMeldSets(cards);
            int deadwood = bestMeldSets.isEmpty() ? 
                GinRummyUtil.getDeadwoodPoints(cards) : 
                GinRummyUtil.getDeadwoodPoints(bestMeldSets.get(0), cards);
                
            collector.recordGameState(
                gameId,
                turnNumber,
                playerNum,
                cards,
                knownOpponentCards,
                faceUpCard,
                discardPile,
                action,
                reward,
                gameOver,
                deadwood
            );
        }
    }
    
    @Override
    public void reportScores(int[] scores) {
        if (collector != null && gameEnded) {
            // Record final game state with the actual reward
            double reward = scores[playerNum] > scores[1-playerNum] ? 1.0 : 
                          scores[playerNum] < scores[1-playerNum] ? -1.0 : 0.0;
            recordGameState("game_over", reward, true);
        }
    }
    
    @Override
    public void reportFinalHand(int playerNum, ArrayList<Card> hand) {
        // Not used for data collection
    }
    
    @Override
    public void reportLayoff(int playerNum, Card layoffCard, ArrayList<Card> opponentMeld) {
        // Track opponent cards if they lay off
        if (playerNum != this.playerNum) {
            knownOpponentCards.add(layoffCard);
        }
    }
} 