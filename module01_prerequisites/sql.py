# sql.py
# Explanation of SQL (Structured Query Language)
# SQL is used to communicate with relational databases.
# It allows creating, reading, updating, and deleting data.

import sqlite3  # Import SQLite3 module for database operations

def create_database():  # Define function to create and populate a sample database
    """
    Create an in-memory SQLite database and table for demonstration.
    """
    conn = sqlite3.connect(':memory:')  # Create an in-memory database connection (data exists only in RAM)
    cursor = conn.cursor()  # Create a cursor object to execute SQL commands

    # Create table
    cursor.execute('''
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            age INTEGER,
            email TEXT
        )
    ''')

    # Insert sample data
    sample_data = [  # Define sample data as a list of tuples
        ('Alice', 25, 'alice@example.com'),  # First user record
        ('Bob', 30, 'bob@example.com'),  # Second user record
        ('Charlie', 35, 'charlie@example.com')  # Third user record
    ]
    cursor.executemany('INSERT INTO users (name, age, email) VALUES (?, ?, ?)', sample_data)  # Insert multiple records using parameterized query

    conn.commit()  # Commit the transaction to save changes
    return conn, cursor  # Return connection and cursor for further operations

def execute_query(cursor, query, params=None):  # Define helper function to execute queries
    """
    Execute a SQL query and return results.
    """
    if params:  # Check if parameters are provided for parameterized query
        cursor.execute(query, params)  # Execute query with parameters to prevent SQL injection
    else:  # If no parameters provided
        cursor.execute(query)  # Execute query directly
    return cursor.fetchall()  # Return all results as a list of tuples

# Create database
conn, cursor = create_database()  # Call function to create and populate database

# Examples of SQL operations
print("1. SELECT all users:")  # Print header for first example
results = execute_query(cursor, "SELECT * FROM users")  # Execute SELECT query to get all users
for row in results:  # Iterate through each result row
    print(row)  # Print the entire row (tuple)

print("\n2. SELECT users older than 28:")  # Print header for second example
results = execute_query(cursor, "SELECT name, age FROM users WHERE age > ?", (28,))  # Execute SELECT with WHERE clause and parameter
for row in results:  # Iterate through filtered results
    print(row)  # Print name and age of users older than 28

print("\n3. UPDATE user age:")  # Print header for third example
cursor.execute("UPDATE users SET age = ? WHERE name = ?", (26, 'Alice'))  # Execute UPDATE query to change Alice's age
conn.commit()  # Commit the update transaction
results = execute_query(cursor, "SELECT name, age FROM users WHERE name = 'Alice'")  # Query to verify the update
print(results)  # Print the updated record

print("\n4. DELETE a user:")  # Print header for fourth example
cursor.execute("DELETE FROM users WHERE name = 'Charlie'")  # Execute DELETE query to remove Charlie
conn.commit()  # Commit the delete transaction
results = execute_query(cursor, "SELECT COUNT(*) FROM users")  # Count remaining users
print(f"Remaining users: {results[0][0]}")  # Print the count of remaining users

# Close connection
conn.close()  # Close the database connection to free resources

# SQL concepts:
# - DDL: CREATE, ALTER, DROP (Data Definition Language)
# - DML: SELECT, INSERT, UPDATE, DELETE (Data Manipulation Language)
# - DCL: GRANT, REVOKE (Data Control Language)
# - TCL: COMMIT, ROLLBACK (Transaction Control Language)