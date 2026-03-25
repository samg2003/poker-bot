export const formatCard = (c) => {
  if (c === -1) return '';
  const suits = ['♣', '♦', '♥', '♠'];
  const ranks = ['2','3','4','5','6','7','8','9','T','J','Q','K','A'];
  const suit = suits[Math.floor(c / 13)];
  const rank = ranks[c % 13];
  return `${rank}${suit}`;
}
