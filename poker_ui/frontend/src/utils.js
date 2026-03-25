export const formatCard = (c) => {
  if (c === -1) return '';
  const suits = ['♣', '♦', '♥', '♠'];
  const ranks = ['2','3','4','5','6','7','8','9','T','J','Q','K','A'];
  const rank = Math.floor(c / 4);
  const suit = c % 4;
  return `${ranks[rank]}${suits[suit]}`;
}

// Diamonds (1) and Hearts (2) are red
export const isRedSuit = (c) => {
  if (c === -1) return false;
  const suit = c % 4;
  return suit === 1 || suit === 2;
}
