/**
 * @typedef {Object} Message
 * @property {string} id - Identifiant unique du message
 * @property {string} content - Contenu du message
 * @property {'user' | 'bot'} sender - Expéditeur du message
 * @property {Date} timestamp - Horodatage du message
 * @property {'text' | 'image' | 'file'} type - Type de message
 * @property {'sent' | 'delivered' | 'read'} [status] - Statut du message (optionnel)
 */

/**
 * @typedef {Object} ChatSettings
 * @property {'light' | 'dark'} theme - Thème de l'interface
 * @property {'small' | 'medium' | 'large'} fontSize - Taille de police
 * @property {boolean} animations - Activation des animations
 * @property {boolean} sounds - Activation des sons
 */

export {};

