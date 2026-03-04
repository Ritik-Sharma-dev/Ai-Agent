import instaloader
import json
from urllib.parse import urlparse

def extract_shortcode(url_or_code: str) -> str:
    if "instagram.com" in url_or_code:
        path = urlparse(url_or_code).path.strip("/")
        parts = path.split("/")
        # Expected: /p/SHORTCODE/
        if len(parts) >= 2 and parts[0] == "p":
            return parts[1]
        raise ValueError("Invalid Instagram post URL")
    return url_or_code.strip()

def main():
    user_input = input("Enter Instagram post URL or shortcode: ").strip()
    shortcode = extract_shortcode(user_input)

    print("Target shortcode:", shortcode)

    L = instaloader.Instaloader(
        download_pictures=False,
        download_videos=False,
        download_video_thumbnails=False,
        save_metadata=False,
        compress_json=False,
        quiet=False,
    )

    try:
        print("Fetching post (public, no login)...")
        post = instaloader.Post.from_shortcode(L.context, shortcode)
    except Exception as e:
        print("❌ Failed to fetch post.")
        print("Reason:", e)
        print("⚠️ If you see 429 or useragent mismatch, wait 30–60 minutes or change network.")
        return

    result = {
        "shortcode": post.shortcode,
        "url": f"https://www.instagram.com/p/{post.shortcode}/",
        "comments": []
    }

    try:
        for comment in post.get_comments():
            comment_obj = {
                "comment_id": comment.id,
                "username": comment.owner.username if comment.owner else None,
                "text": comment.text,
                "likes": comment.likes_count,
                "created_at": comment.created_at_utc.isoformat(),
                "replies": []
            }

            for reply in comment.answers:
                reply_obj = {
                    "comment_id": reply.id,
                    "username": reply.owner.username if reply.owner else None,
                    "text": reply.text,
                    "likes": reply.likes_count,
                    "created_at": reply.created_at_utc.isoformat(),
                }
                comment_obj["replies"].append(reply_obj)

            result["comments"].append(comment_obj)

    except Exception as e:
        print("⚠️ Error while fetching comments:", e)

    with open("comments_one_post.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n✅ Done. Saved to comments_one_post.json")

if __name__ == "__main__":
    main()