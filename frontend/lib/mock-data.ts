// Mock data for demo purposes
import type { UserProfile, TasteDimension } from './types'

export const mockDimensions: TasteDimension[] = [
  { key: 'sweet_leaning', label: 'Sweet', value: 72, description: 'Preference for sweet flavors' },
  { key: 'salty_leaning', label: 'Salty', value: 58, description: 'Preference for savory, salty dishes' },
  { key: 'umami_leaning', label: 'Umami', value: 85, description: 'Love for deep, savory richness' },
  { key: 'spicy_leaning', label: 'Spice', value: 45, description: 'Heat tolerance and preference' },
  { key: 'richness_preference', label: 'Richness', value: 78, description: 'Preference for rich, indulgent foods' },
  { key: 'freshness_preference', label: 'Fresh', value: 62, description: 'Affinity for fresh, light flavors' },
  { key: 'texture_seeking', label: 'Texture', value: 70, description: 'Interest in varied textures' },
  { key: 'comfort_food_tendency', label: 'Comfort', value: 82, description: 'Gravitation toward comfort foods' },
  { key: 'adventurousness', label: 'Adventure', value: 65, description: 'Willingness to try new foods' },
  { key: 'variety_seeking', label: 'Variety', value: 73, description: 'Desire for diverse cuisines' },
  { key: 'protein_forward', label: 'Protein', value: 68, description: 'Preference for protein-rich dishes' },
  { key: 'carb_forward', label: 'Carbs', value: 75, description: 'Love for carb-heavy dishes' },
  { key: 'dessert_affinity', label: 'Dessert', value: 88, description: 'Sweet tooth intensity' },
  { key: 'global_cuisine_breadth', label: 'Global', value: 71, description: 'Range of cuisines explored' },
]

export const mockProfile: UserProfile = {
  username: 'foodie_sarah',
  archetype: 'Fries Non-Negotiable',
  archetype_description: 'You pick comfort food that actually comforts: crispy, salty, and not pretending. You\'ve never really been against fries. Just briefly unavailable.',
  archetype_graphic: 'assets/archetypes/fries_non_negotiable.png',
  joke: 'Your ideal date night involves a heated debate about the optimal cheese-to-pasta ratio.',
  observations: [
    'You show a strong preference for umami-rich dishes, especially those with aged cheeses and slow-cooked meats.',
    'Desserts are definitely your thing — your sweet tooth is well above average.',
    'You balance comfort food cravings with a genuine appreciation for fresh, vibrant flavors.',
    'Your cuisine range is impressively broad, suggesting a curious and open palate.',
    'Interestingly, you seem to prefer medium spice levels, enjoying heat as an accent rather than the main event.',
  ],
  favorite_cuisines: {
    Italian: 4.2,
    Japanese: 3.4,
    Mexican: 2.1,
  },
  favorite_dishes: {
    carbonara: 3.0,
    'tonkotsu ramen': 2.8,
    tiramisu: 2.5,
  },
  favorite_traits: {
    'comfort-food': 3.4,
    'dessert-leaning': 2.8,
    'protein-forward': 2.0,
  },
  taste_profile: {
    dimensions: mockDimensions,
    analysis: 'Your flavor fingerprint reveals a palate that seeks satisfaction in every bite. You have a pronounced love for umami and richness, suggesting you gravitate toward dishes with depth and complexity. Your high dessert affinity paired with moderate freshness preference indicates you enjoy contrast — the sweet after the savory, the light after the rich. Your adventurousness score shows you\'re open to new experiences while still maintaining favorite comfort zones.',
    history: [],
    relative_rankings: [
      { label: 'More comfort-oriented', description: 'Than 78% of users', percentile: 78, direction: 'higher' },
      { label: 'Higher dessert affinity', description: 'Than 85% of users', percentile: 85, direction: 'higher' },
      { label: 'Broader cuisine range', description: 'Than 67% of users', percentile: 67, direction: 'higher' },
      { label: 'More umami-seeking', description: 'Than 82% of users', percentile: 82, direction: 'higher' },
      { label: 'Similar spice tolerance', description: 'To most users', percentile: 52, direction: 'similar' },
    ],
  },
  compatible_users: [
    { username: 'pasta_lover_mike', compatibility_score: 94, reason: 'You both keep ending up in the same comfort-food lane.' },
    { username: 'chef_elena', compatibility_score: 87, reason: 'You have a similar mix of curiosity and reliable favorites.' },
    { username: 'brunch_queen', compatibility_score: 82, reason: 'Your food habits overlap in a way that feels pretty natural.' },
    { username: 'spice_explorer_jay', compatibility_score: 76, reason: 'You are drawn to similar kinds of meals, just with your own twist.' },
  ],
  clicked_recommendations: [
    {
      event_id: 'rec_1',
      dish_label: 'Cacio E Pepe',
      cuisine: 'Italian',
      signal_weight: 0.22,
      timestamp: '2024-01-16T12:00:00Z',
    },
  ],
  recent_uploads: [
    { id: '1', image_url: '/uploads/carbonara.jpg', dish_label: 'Carbonara', cuisine: 'Italian', tags: ['pasta', 'creamy', 'comfort'], uploaded_at: '2024-01-15' },
    { id: '2', image_url: '/uploads/ramen.jpg', dish_label: 'Tonkotsu Ramen', cuisine: 'Japanese', tags: ['soup', 'umami', 'noodles'], uploaded_at: '2024-01-14' },
    { id: '3', image_url: '/uploads/tiramisu.jpg', dish_label: 'Tiramisu', cuisine: 'Italian', tags: ['dessert', 'coffee', 'sweet'], uploaded_at: '2024-01-13' },
    { id: '4', image_url: '/uploads/tacos.jpg', dish_label: 'Birria Tacos', cuisine: 'Mexican', tags: ['tacos', 'beef', 'spicy'], uploaded_at: '2024-01-12' },
    { id: '5', image_url: '/uploads/croissant.jpg', dish_label: 'Almond Croissant', cuisine: 'French', tags: ['pastry', 'sweet', 'breakfast'], uploaded_at: '2024-01-11' },
    { id: '6', image_url: '/uploads/pho.jpg', dish_label: 'Beef Pho', cuisine: 'Vietnamese', tags: ['soup', 'noodles', 'aromatic'], uploaded_at: '2024-01-10' },
  ],
  total_uploads: 47,
  created_at: '2023-11-20',
}

export const emptyProfile: UserProfile = {
  username: 'new_user',
  archetype: '',
  archetype_description: '',
  archetype_graphic: '',
  joke: '',
  observations: [],
  favorite_cuisines: {},
  favorite_dishes: {},
  favorite_traits: {},
  taste_profile: {
    dimensions: [],
    analysis: '',
    history: [],
    relative_rankings: [],
  },
  compatible_users: [],
  clicked_recommendations: [],
  recent_uploads: [],
  total_uploads: 0,
  created_at: new Date().toISOString(),
}
