import csv
import pathlib
import os


def generate_strategy_csv(policy, filename_prefix="strategy", output_dir="strategyTable"):
    """
    Generate CSV files representing the learned blackjack strategy.

    Parameters:
    - policy: dict mapping state -> action,
      where state is a tuple: (player_total, dealer_card, usable_ace)
    - filename_prefix: prefix for the output CSV filenames.
    
    This function creates:
    - 'strategy_hard.csv': for states with no usable ace.
    - 'strategy_soft.csv': for states with a usable ace.
    """
    
    dealer_cards = list(range(1, 11))  # Dealer's up-card: 1 (Ace) through 10.
    player_totals = list(range(4, 22))   # Possible player totals (from 4 to 21).

    # Generate CSV for Hard Totals (no usable ace)
    hard_filename = os.path.join(output_dir,f"{filename_prefix}_hard.csv") 
    with open(hard_filename, mode="w", newline="") as hard_file:
        writer = csv.writer(hard_file)
        # Header row: first column label then dealer cards.
        header = ["Player Total \\ Dealer"] + dealer_cards
        writer.writerow(header)
        
        for total in player_totals:
            row = [total]
            for dealer in dealer_cards:
                # State representation: (player_total, dealer_card, usable_ace=False)
                state = (total, dealer, False)
                action = policy.get(state, "N/A")
                row.append(action)
            writer.writerow(row)

    # Generate CSV for Soft Totals (with usable ace)
    soft_filename = os.path.join(output_dir,f"{filename_prefix}_soft.csv")
    with open(soft_filename, mode="w", newline="") as soft_file:
        writer = csv.writer(soft_file)
        header = ["Player Total \\ Dealer"] + dealer_cards
        writer.writerow(header)
        
        for total in player_totals:
            row = [total]
            for dealer in dealer_cards:
                state = (total, dealer, True)
                action = policy.get(state, "N/A")
                row.append(action)
            writer.writerow(row)

    print(f"CSV strategy files generated: {hard_filename} and {soft_filename}")


# Example: Using a dummy learned policy for demonstration.
# In practice, your policy would be the output of your RL training.
dummy_policy = {
    # For example, basic strategy: if player total >= 17, stand; otherwise hit.
    # This is oversimplified and just for demonstration.
    (total, dealer, usable) : ("stand" if total >= 17 else "hit")
    for total in range(4, 22)
    for dealer in range(1, 11)
    for usable in [False, True]
}

# Generate CSV files from the dummy policy:
generate_strategy_csv(dummy_policy, filename_prefix="dummy_strategy")
