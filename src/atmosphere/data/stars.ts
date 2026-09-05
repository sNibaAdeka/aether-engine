/**
 * The bright-star catalogue: real J2000 coordinates for the stars that make the
 * constellations.
 *
 * Spec 3.3 asks for "процедурные, по реальному каталогу ярких звёзд (можно
 * захардкодить 200 самых ярких с координатами)". The reason real coordinates
 * matter is not astronomy pedantry — it is that a random star field has no
 * shapes in it, and the eye finds Orion, the Plough and Cassiopeia instantly.
 * A sky without them reads as noise; a sky with them reads as *the* sky.
 *
 * Layout is a flat Float32Array, four numbers per star:
 *   [0] right ascension, hours (0–24)
 *   [1] declination, degrees
 *   [2] apparent visual magnitude
 *   [3] B−V colour index — the star's colour, −0.3 (blue) to +1.9 (deep orange)
 *
 * Flat rather than an array of objects, per CLAUDE.md rule 2: this is bulk data
 * that goes straight into a GPU buffer.
 *
 * TODO(quality): these coordinates are written from knowledge rather than
 * ingested from a machine-readable Yale BSC. The famous few dozen are good to
 * about an arcminute; the fainter half of the list is good to a few tenths of a
 * degree, which is below the resolution the star sprites are drawn at but is not
 * catalogue accuracy. If a constellation ever needs to be identified in-game to
 * better than that, this table should be generated from the real BSC5 rather
 * than corrected by hand.
 */

/* eslint-disable prettier/prettier */
export const STAR_DATA = new Float32Array([
  // ---- first magnitude and brighter -------------------------------------
  6.7525, -16.716, -1.46, 0.00, // Sirius
  6.3992, -52.696, -0.74, 0.15, // Canopus
  14.6600, -60.833, -0.27, 0.71, // Rigil Kentaurus
  14.2610, 19.182, -0.05, 1.23, // Arcturus
  18.6156, 38.784, 0.03, 0.00, // Vega
  5.2782, 45.998, 0.08, 0.80, // Capella
  5.2423, -8.202, 0.13, -0.03, // Rigel
  7.6550, 5.225, 0.34, 0.42, // Procyon
  1.6286, -57.237, 0.46, -0.16, // Achernar
  5.9195, 7.407, 0.50, 1.85, // Betelgeuse
  14.0637, -60.373, 0.61, -0.23, // Hadar
  19.8464, 8.868, 0.77, 0.22, // Altair
  12.4433, -63.099, 0.77, -0.24, // Acrux
  4.5987, 16.509, 0.85, 1.54, // Aldebaran
  16.4901, -26.432, 0.96, 1.83, // Antares
  13.4199, -11.161, 1.04, -0.23, // Spica
  7.7553, 28.026, 1.14, 1.00, // Pollux
  22.9608, -29.622, 1.16, 0.09, // Fomalhaut
  20.6905, 45.280, 1.25, 0.09, // Deneb
  12.7953, -59.689, 1.25, -0.24, // Mimosa
  10.1395, 11.967, 1.35, -0.11, // Regulus
  6.9770, -28.972, 1.50, -0.21, // Adhara
  7.5767, 31.888, 1.58, 0.03, // Castor
  17.5601, -37.104, 1.62, -0.22, // Shaula
  12.5194, -57.113, 1.63, 1.59, // Gacrux
  5.4188, 6.350, 1.64, -0.22, // Bellatrix
  5.4382, 28.608, 1.65, -0.13, // Elnath
  9.2200, -69.717, 1.67, 0.07, // Miaplacidus
  5.6036, -1.202, 1.69, -0.18, // Alnilam
  22.1372, -46.961, 1.74, -0.13, // Alnair
  5.6793, -1.943, 1.74, -0.20, // Alnitak
  12.9005, 55.960, 1.77, -0.02, // Alioth
  11.0621, 61.751, 1.79, 1.07, // Dubhe
  3.4054, 49.861, 1.79, 0.48, // Mirfak
  7.1399, -26.393, 1.83, 0.67, // Wezen
  8.1584, -47.337, 1.83, -0.18, // Muhlifain
  18.4029, -34.385, 1.85, -0.03, // Kaus Australis
  13.7923, 49.313, 1.86, -0.19, // Alkaid
  17.6220, -42.998, 1.86, 0.40, // Sargas
  8.3752, -59.510, 1.86, 1.19, // Avior
  5.9922, 44.947, 1.90, 0.08, // Menkalinan
  16.8110, -69.028, 1.91, 1.44, // Atria
  6.6285, 16.399, 1.93, 0.00, // Alhena
  20.4275, -56.735, 1.94, -0.12, // Peacock
  8.7450, -54.709, 1.96, 0.04, // Delta Velorum
  2.5303, 89.264, 1.98, 0.60, // Polaris
  6.3783, -17.956, 1.98, -0.24, // Mirzam
  9.4597, -8.659, 2.00, 1.44, // Alphard
  2.1195, 23.462, 2.00, 1.15, // Hamal
  10.3329, 19.841, 2.01, 1.13, // Algieba

  // ---- second magnitude ---------------------------------------------------
  0.7264, -17.987, 2.04, 1.02, // Diphda
  18.9211, -26.297, 2.05, -0.22, // Nunki
  0.1398, 29.091, 2.06, -0.11, // Alpheratz
  5.7959, -9.670, 2.06, -0.17, // Saiph
  14.1114, -36.370, 2.06, 1.01, // Menkent
  1.1622, 35.621, 2.06, 1.58, // Mirach
  14.8451, 74.156, 2.08, 1.47, // Kochab
  17.5822, 12.560, 2.08, 0.16, // Rasalhague
  3.1361, 40.956, 2.09, -0.05, // Algol
  2.0650, 42.330, 2.10, 1.37, // Almach
  22.7113, -46.885, 2.11, 1.60, // Beta Gruis
  11.8177, 14.572, 2.14, 0.09, // Denebola
  0.9451, 60.717, 2.15, -0.15, // Cih
  12.6919, -48.960, 2.17, -0.01, // Gamma Centauri
  8.0597, -40.003, 2.21, -0.27, // Naos
  9.2850, -59.275, 2.21, 0.18, // Aspidiske
  15.5781, 26.715, 2.22, -0.02, // Alphecca
  9.1332, -43.433, 2.23, 1.66, // Suhail
  5.5334, -0.299, 2.23, -0.18, // Mintaka
  20.3705, 40.257, 2.23, 0.68, // Sadr
  17.9435, 51.489, 2.23, 1.52, // Eltanin
  13.3988, 54.925, 2.23, 0.02, // Mizar
  0.6751, 56.537, 2.24, 1.17, // Schedar
  0.1530, 59.150, 2.27, 0.34, // Caph
  16.0055, -22.622, 2.29, -0.12, // Dschubba
  16.8360, -34.293, 2.29, 1.14, // Larawag
  14.6989, -47.388, 2.30, -0.20, // Alpha Lupi
  13.6647, -53.466, 2.30, -0.22, // Epsilon Centauri
  14.5951, -42.158, 2.31, -0.19, // Eta Centauri
  11.0307, 56.382, 2.34, 0.03, // Merak
  14.7498, 27.074, 2.35, 0.97, // Izar
  21.7364, 9.875, 2.38, 1.53, // Enif
  17.7082, -39.030, 2.39, -0.20, // Girtab
  0.4381, -42.306, 2.40, 1.09, // Ankaa
  11.8972, 53.695, 2.41, 0.04, // Phecda
  23.0629, 28.083, 2.42, 1.67, // Scheat
  17.1729, -15.725, 2.43, 0.06, // Sabik
  7.4014, -29.303, 2.45, -0.08, // Aludra
  21.3097, 62.586, 2.45, 0.22, // Alderamin
  23.0793, 15.205, 2.48, -0.04, // Markab
  20.7702, 33.970, 2.48, 1.02, // Gienah Cygni
  9.3680, -55.011, 2.50, -0.14, // Kappa Velorum
  3.0380, 4.090, 2.53, 1.63, // Menkar
  16.6191, -10.567, 2.54, 0.02, // Han
  13.9257, -47.288, 2.55, -0.22, // Zeta Centauri
  5.5455, -17.822, 2.58, 0.21, // Arneb
  12.1397, -50.722, 2.58, -0.12, // Delta Centauri
  19.0436, -29.880, 2.60, 0.08, // Ascella
  15.2830, -9.383, 2.61, -0.11, // Zubeneschamali
  16.0906, -19.805, 2.62, -0.07, // Acrab
  15.7378, 6.426, 2.63, 1.17, // Unukalhai
  1.9105, 20.808, 2.64, 0.13, // Sheratan
  1.4303, 60.235, 2.68, 0.13, // Ruchbah
  13.9114, 18.398, 2.68, 0.58, // Muphrid
  14.9758, -43.134, 2.68, -0.21, // Beta Lupi
  12.6194, -69.136, 2.69, -0.20, // Alpha Muscae
  17.5121, -37.296, 2.69, -0.22, // Lesath
  18.3499, -29.828, 2.70, 1.38, // Kaus Media
  19.7709, 10.613, 2.72, 1.52, // Tarazed
  16.2393, -3.694, 2.73, 1.58, // Yed Prior
  12.6943, -1.449, 2.74, 0.36, // Porrima
  14.8479, -16.042, 2.75, 0.15, // Zubenelgenubi
  13.3379, -36.712, 2.75, 0.06, // Iota Centauri
  10.7150, -64.394, 2.76, -0.22, // Theta Carinae
  16.5036, 21.490, 2.77, 0.94, // Kornephoros
  15.5850, -41.167, 2.77, -0.19, // Gamma Lupi
  0.4294, -77.254, 2.78, 0.62, // Beta Hydri
  12.2525, -58.749, 2.79, -0.19, // Delta Crucis
  5.1308, -5.086, 2.79, 0.13, // Cursa
  17.5071, 52.301, 2.79, 0.95, // Rastaban
  16.6882, 31.603, 2.81, 0.65, // Zeta Herculis
  18.4661, -25.422, 2.81, 1.04, // Kaus Borealis
  13.0362, 10.959, 2.83, 0.94, // Vindemiatrix
  0.2206, 15.184, 2.83, -0.23, // Algenib
  5.4706, -20.759, 2.84, 0.82, // Nihal
  21.7840, -16.127, 2.85, 0.29, // Deneb Algedi
  17.5307, -49.876, 2.85, -0.17, // Alpha Arae
  17.4212, -55.530, 2.85, 1.46, // Beta Arae
  15.9192, -63.431, 2.85, 0.29, // Beta Trianguli Australis
  22.3083, -60.260, 2.86, 1.39, // Alpha Tucanae
  1.9799, -61.570, 2.86, 0.28, // Alpha Hydri
  3.7914, 24.105, 2.87, -0.09, // Alcyone
  19.7495, 45.131, 2.87, -0.05, // Delta Cygni
  2.9710, -40.305, 2.88, 0.13, // Acamar
  15.3153, -68.679, 2.89, 0.01, // Gamma Trianguli Australis
  19.1621, -21.024, 2.89, 0.35, // Albaldah
  21.5257, -5.571, 2.90, 0.83, // Sadalsuud
  22.6996, 30.221, 2.93, 0.86, // Matar
  3.9678, -13.509, 2.95, 1.59, // Zaurak
  22.0964, -0.320, 2.95, 0.98, // Sadalmelik
  9.7850, -65.072, 2.97, 0.27, // Upsilon Carinae
  18.0966, -30.424, 2.98, 0.98, // Alnasl

  // ---- third magnitude: the stars that close the constellation figures ----
  5.6275, 21.143, 3.00, -0.19, // Zeta Tauri
  21.8988, -37.365, 3.00, -0.12, // Gamma Gruis
  2.1595, 34.987, 3.00, 0.14, // Beta Trianguli
  14.5340, 38.308, 3.03, 0.19, // Gamma Bootis
  20.3501, -14.781, 3.05, 0.79, // Dabih
  19.5121, 27.960, 3.05, 1.09, // Albireo
  15.3455, 71.834, 3.05, 0.05, // Pherkad
  19.2093, 67.662, 3.07, 1.00, // Altais
  20.6262, -47.291, 3.11, 1.00, // Alpha Indi
  17.2503, 24.839, 3.12, 0.08, // Delta Herculis
  17.2506, 36.809, 3.16, 1.44, // Pi Herculis
  14.7083, -64.975, 3.19, 0.26, // Alpha Circini
  23.6558, 77.632, 3.21, 1.03, // Errai
  21.4776, 70.561, 3.23, -0.22, // Alphirk
  18.9824, 32.690, 3.24, -0.05, // Sulafat
  22.9109, -15.821, 3.27, 0.05, // Skat
  0.6556, 30.861, 3.27, 1.28, // Delta Andromedae
  15.4155, 58.966, 3.29, 1.16, // Edasich
  12.2571, 57.033, 3.31, 0.08, // Megrez
  19.4249, 3.115, 3.36, 0.32, // Delta Aquilae
  17.2443, 14.390, 3.35, 1.44, // Rasalgethi
  1.9065, 63.670, 3.35, -0.15, // Segin
  5.5855, 9.934, 3.39, -0.16, // Meissa
  22.6910, 10.832, 3.40, -0.09, // Zeta Pegasi
  1.8846, 29.579, 3.41, 0.49, // Alpha Trianguli
  20.7503, -66.203, 3.42, 0.16, // Beta Pavonis
  15.2582, 33.315, 3.47, 0.95, // Delta Bootis
  15.0322, 40.390, 3.49, 0.97, // Nekkar
  18.8347, 33.363, 3.52, 0.00, // Sheliak
  16.7147, 38.922, 3.53, 0.92, // Eta Herculis
  20.3003, -12.545, 3.57, 0.89, // Algedi
  12.3564, -60.401, 3.59, 1.42, // Epsilon Crucis
  14.0731, 64.376, 3.65, -0.05, // Thuban
  21.6683, -16.662, 3.68, 0.32, // Nashira
  19.9219, 6.407, 3.71, 0.86, // Alshain
  20.6606, 15.912, 3.77, -0.06, // Sualocin
  2.0334, 2.764, 3.82, 0.32, // Alrescha
  19.3982, -40.616, 3.97, -0.10, // Rukbat
  21.1467, -88.956, 5.47, 0.26, // Sigma Octantis — the south pole star
]);

/** Number of catalogued stars. */
export const STAR_COUNT = STAR_DATA.length / 4;

/**
 * Galactic north pole in J2000 equatorial coordinates, and the direction of the
 * galactic centre. Both are needed to place the Milky Way, which is a band
 * around the galactic equator brightening sharply toward Sagittarius.
 */
export const GALACTIC_POLE_RA_HOURS = 12.85694; // 12h 51m 26s
export const GALACTIC_POLE_DEC_DEG = 27.128;
export const GALACTIC_CENTRE_RA_HOURS = 17.7611; // 17h 45m 40s
export const GALACTIC_CENTRE_DEC_DEG = -28.936;
