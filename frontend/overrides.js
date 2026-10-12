window.OVERRIDES = {
  "phrases": {
    "kal 20 croissant ka order hai": {
      "item": "CROISSANT",
      "delta_units": 20,
      "confidence": 0.95,
      "reason": "Catering order for 20 croissants tomorrow",
      "source": "seeded"
    },
    "shaadi hai saturday thoda zyada bana lena": {
      "item": "",
      "delta_units": 0,
      "confidence": 0.4,
      "reason": "Wedding Saturday, quantity not specified",
      "source": "seeded"
    },
    "kal 80 puff ka order hai": {
      "item": "CROISSANT",
      "delta_units": 80,
      "confidence": 0.55,
      "reason": "Catering order for 80 puffs tomorrow; nearest item on this menu is croissant",
      "source": "seeded"
    }
  },
  "fallback": "kal 20 croissant ka order hai"
};
