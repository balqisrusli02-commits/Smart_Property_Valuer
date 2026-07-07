-- LRGS FTSM UKM: use the faculty-assigned database name
USE a202336;

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(120) NULL,
    contact_number VARCHAR(30) NULL,
    address TEXT NULL,
    postcode VARCHAR(10) NULL,
    state VARCHAR(100) NULL,
    marital_status VARCHAR(20) NULL,
    family_count INT NULL,
    profile_image_url VARCHAR(1200) NULL,
    profile_image_blob LONGBLOB NULL,
    profile_image_mime VARCHAR(80) NULL,
    profile_image_name VARCHAR(255) NULL,
    user_role VARCHAR(20) NOT NULL DEFAULT 'user',
    profile_theme VARCHAR(40) NOT NULL DEFAULT 'neon-purple',
    avatar_size INT NOT NULL DEFAULT 120,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_users_email (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

ALTER TABLE users
ADD COLUMN IF NOT EXISTS profile_image_url VARCHAR(1200) NULL;

ALTER TABLE users
ADD COLUMN IF NOT EXISTS profile_image_blob LONGBLOB NULL;

ALTER TABLE users
ADD COLUMN IF NOT EXISTS profile_image_mime VARCHAR(80) NULL;

ALTER TABLE users
ADD COLUMN IF NOT EXISTS profile_image_name VARCHAR(255) NULL;

ALTER TABLE users
ADD COLUMN IF NOT EXISTS user_role VARCHAR(20) NOT NULL DEFAULT 'user';

CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    input_json LONGTEXT NOT NULL,
    predicted_price DECIMAL(15,2) NOT NULL,
    user_id INT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_prediction_created_at (created_at),
    INDEX idx_prediction_price (predicted_price),
    INDEX idx_predictions_user_id (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS password_reset_requests (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NULL,
    customer_name VARCHAR(120) NULL,
    email VARCHAR(255) NOT NULL,
    reason TEXT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'Pending',
    request_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    admin_action_date TIMESTAMP NULL,
    admin_action_by VARCHAR(255) NULL,
    completed_at TIMESTAMP NULL,
    INDEX idx_password_reset_email (email),
    INDEX idx_password_reset_status (status),
    INDEX idx_password_reset_date (request_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

ALTER TABLE password_reset_requests
ADD COLUMN IF NOT EXISTS customer_name VARCHAR(120) NULL;

ALTER TABLE password_reset_requests
ADD COLUMN IF NOT EXISTS admin_action_date TIMESTAMP NULL;

ALTER TABLE password_reset_requests
ADD COLUMN IF NOT EXISTS admin_action_by VARCHAR(255) NULL;

ALTER TABLE password_reset_requests
ADD COLUMN IF NOT EXISTS completed_at TIMESTAMP NULL;

CREATE TABLE IF NOT EXISTS property_listings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    area VARCHAR(100) NOT NULL,
    negeri VARCHAR(100) NOT NULL,
    property_type INT NOT NULL,
    built_up_sf DECIMAL(10,2) NOT NULL,
    land_size DECIMAL(10,2) NOT NULL,
    bedroom INT NOT NULL,
    bathroom INT NOT NULL,
    car_park INT NOT NULL,
    furnishing INT NOT NULL,
    tenure INT NOT NULL,
    unit_type INT NOT NULL,
    listing_price DECIMAL(15,2) NOT NULL,
    latitude DECIMAL(10,6) NULL,
    longitude DECIMAL(10,6) NULL,
    description TEXT NULL,
    image_url VARCHAR(1200) NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_property_state (negeri),
    INDEX idx_property_price (listing_price),
    INDEX idx_property_type (property_type),
    INDEX idx_property_bedroom (bedroom),
    INDEX idx_property_state_type_bedroom (negeri, property_type, bedroom),
    INDEX idx_property_created_at (created_at),
    INDEX idx_property_geo (latitude, longitude)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

ALTER TABLE property_listings
ADD COLUMN IF NOT EXISTS image_url VARCHAR(1200) NULL;

CREATE INDEX IF NOT EXISTS idx_prediction_created_at
ON predictions(created_at);

CREATE INDEX IF NOT EXISTS idx_prediction_price
ON predictions(predicted_price);

CREATE INDEX IF NOT EXISTS idx_property_state
ON property_listings(negeri);

CREATE INDEX IF NOT EXISTS idx_property_price
ON property_listings(listing_price);

CREATE INDEX IF NOT EXISTS idx_property_type
ON property_listings(property_type);

CREATE INDEX IF NOT EXISTS idx_property_bedroom
ON property_listings(bedroom);

CREATE INDEX IF NOT EXISTS idx_property_state_type_bedroom
ON property_listings(negeri, property_type, bedroom);

CREATE INDEX IF NOT EXISTS idx_property_created_at
ON property_listings(created_at);

CREATE INDEX IF NOT EXISTS idx_property_geo
ON property_listings(latitude, longitude);

CREATE OR REPLACE VIEW vw_state_sales_distribution AS
SELECT
    negeri AS state_name,
    COUNT(*) AS total_listings
FROM property_listings
WHERE negeri IS NOT NULL
  AND negeri <> ''
GROUP BY negeri;

CREATE OR REPLACE VIEW vw_top_state_sales AS
SELECT
    state_name,
    total_listings
FROM vw_state_sales_distribution
ORDER BY total_listings DESC, state_name ASC
LIMIT 1;

CREATE OR REPLACE VIEW vw_latest_predictions AS
SELECT
    id,
    predicted_price,
    created_at,
    input_json
FROM predictions
ORDER BY created_at DESC;
