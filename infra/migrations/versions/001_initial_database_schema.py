"""Initial database schema

Revision ID: 001
Revises: 
Create Date: 2025-01-08 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create users table for authentication
    op.create_table('users',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('username', sa.String(length=50), nullable=False),
        sa.Column('email', sa.String(length=100), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=100), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('is_superuser', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('last_login', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('username'),
        sa.UniqueConstraint('email')
    )

    # Create user_sessions table
    op.create_table('user_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('session_token', sa.String(length=255), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('user_agent', sa.String(length=500), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('session_token')
    )

    # Create projects table
    op.create_table('projects',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('input_folder', sa.Text(), nullable=False),
        sa.Column('output_folder', sa.Text(), nullable=False),
        sa.Column('performance_mode', sa.String(length=20), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=True),
        sa.Column('settings', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    # Create processing_sessions table
    op.create_table('processing_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('project_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('total_images', sa.Integer(), nullable=False),
        sa.Column('processed_images', sa.Integer(), nullable=True),
        sa.Column('approved_images', sa.Integer(), nullable=True),
        sa.Column('rejected_images', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=True),
        sa.Column('start_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('end_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('session_config', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create image_results table
    op.create_table('image_results',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('image_path', sa.Text(), nullable=False),
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('source_folder', sa.String(length=10), nullable=True),
        sa.Column('quality_scores', sa.JSON(), nullable=True),
        sa.Column('defect_results', sa.JSON(), nullable=True),
        sa.Column('similarity_group', sa.Integer(), nullable=True),
        sa.Column('similar_images', sa.JSON(), nullable=True),
        sa.Column('compliance_results', sa.JSON(), nullable=True),
        sa.Column('final_decision', sa.String(length=20), nullable=False),
        sa.Column('rejection_reasons', postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column('human_override', sa.Boolean(), nullable=True),
        sa.Column('human_review_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('processing_time', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['session_id'], ['processing_sessions.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create checkpoints table
    op.create_table('checkpoints',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('checkpoint_type', sa.String(length=20), nullable=False),
        sa.Column('processed_count', sa.Integer(), nullable=False),
        sa.Column('current_batch', sa.Integer(), nullable=True),
        sa.Column('current_image_index', sa.Integer(), nullable=True),
        sa.Column('session_state', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['session_id'], ['processing_sessions.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create similarity_groups table
    op.create_table('similarity_groups',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('group_hash', sa.String(length=64), nullable=False),
        sa.Column('representative_image', sa.Text(), nullable=True),
        sa.Column('image_count', sa.Integer(), nullable=True),
        sa.Column('similarity_threshold', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['session_id'], ['processing_sessions.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create processing_logs table
    op.create_table('processing_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('level', sa.String(length=10), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('context', sa.JSON(), nullable=True),
        sa.Column('image_path', sa.Text(), nullable=True),
        sa.Column('processing_step', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['session_id'], ['processing_sessions.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create system_metrics table
    op.create_table('system_metrics',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('metric_type', sa.String(length=50), nullable=False),
        sa.Column('metric_value', sa.Float(), nullable=False),
        sa.Column('metric_unit', sa.String(length=20), nullable=True),
        sa.Column('context', sa.JSON(), nullable=True),
        sa.Column('recorded_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['session_id'], ['processing_sessions.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create user_preferences table
    op.create_table('user_preferences',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('preference_key', sa.String(length=100), nullable=False),
        sa.Column('preference_value', sa.JSON(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('preference_key')
    )

    # Create indexes for better performance
    op.create_index('ix_users_username', 'users', ['username'])
    op.create_index('ix_users_email', 'users', ['email'])
    op.create_index('ix_user_sessions_user_id', 'user_sessions', ['user_id'])
    op.create_index('ix_user_sessions_session_token', 'user_sessions', ['session_token'])
    op.create_index('ix_user_sessions_expires_at', 'user_sessions', ['expires_at'])
    op.create_index('ix_projects_status', 'projects', ['status'])
    op.create_index('ix_projects_created_at', 'projects', ['created_at'])
    op.create_index('ix_processing_sessions_project_id', 'processing_sessions', ['project_id'])
    op.create_index('ix_processing_sessions_status', 'processing_sessions', ['status'])
    op.create_index('ix_image_results_session_id', 'image_results', ['session_id'])
    op.create_index('ix_image_results_final_decision', 'image_results', ['final_decision'])
    op.create_index('ix_checkpoints_session_id', 'checkpoints', ['session_id'])
    op.create_index('ix_processing_logs_session_id', 'processing_logs', ['session_id'])
    op.create_index('ix_processing_logs_level', 'processing_logs', ['level'])


def downgrade() -> None:
    # Drop indexes
    op.drop_index('ix_processing_logs_level')
    op.drop_index('ix_processing_logs_session_id')
    op.drop_index('ix_checkpoints_session_id')
    op.drop_index('ix_image_results_final_decision')
    op.drop_index('ix_image_results_session_id')
    op.drop_index('ix_processing_sessions_status')
    op.drop_index('ix_processing_sessions_project_id')
    op.drop_index('ix_projects_created_at')
    op.drop_index('ix_projects_status')
    op.drop_index('ix_user_sessions_expires_at')
    op.drop_index('ix_user_sessions_session_token')
    op.drop_index('ix_user_sessions_user_id')
    op.drop_index('ix_users_email')
    op.drop_index('ix_users_username')
    
    # Drop tables in reverse order
    op.drop_table('user_preferences')
    op.drop_table('system_metrics')
    op.drop_table('processing_logs')
    op.drop_table('similarity_groups')
    op.drop_table('checkpoints')
    op.drop_table('image_results')
    op.drop_table('processing_sessions')
    op.drop_table('projects')
    op.drop_table('user_sessions')
    op.drop_table('users')